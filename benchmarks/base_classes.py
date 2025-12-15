import os
import sys

import torch
from tqdm import tqdm
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLControlNetPipeline,
    T2IAdapter,
    WuerstchenCombinedPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import T5EncoderModel
from diffusers.utils import load_image


sys.path.append(".")

from utils import (  # noqa: E402
    BASE_PATH,
    PROMPT,
    BenchmarkInfo,
    benchmark_fn,
    bytes_to_giga_bytes,
    flush,
    generate_csv_dict,
    write_to_csv,
)

RESOLUTION_MAPPING = {
    "Lykon/DreamShaper": (512, 512),
    "lllyasviel/sd-controlnet-canny": (512, 512),
    "diffusers/controlnet-canny-sdxl-1.0": (1024, 1024),
    "TencentARC/t2iadapter_canny_sd14v1": (512, 512),
    "TencentARC/t2i-adapter-canny-sdxl-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-2-1": (768, 768),
    "stabilityai/stable-diffusion-xl-base-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-xl-refiner-1.0": (1024, 1024),
    "stabilityai/sdxl-turbo": (512, 512),
    "etri-vilab/koala-1b": (1024, 1024),
    "black-forest-labs/FLUX.1-dev": (1024,1024),
    "black-forest-labs/FLUX.1-schnell": (1024,1024),
}


class BaseBenchmak:
    pipeline_class = None

    def __init__(self, args):
        super().__init__()

    def run_inference(self, args):
        raise NotImplementedError

    def benchmark(self, args):
        raise NotImplementedError

    def get_result_filepath(self, args):
        pipeline_class_name = str(self.pipe.__class__.__name__)
        # Include GPU count in filename if multi-GPU
        num_gpus = getattr(args, 'num_gpus', 1)
        gpu_suffix = f"-gpus@{num_gpus}" if num_gpus > 1 else ""
        name = (
            args.ckpt.replace("/", "_")
            + "_"
            + pipeline_class_name
            + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}{gpu_suffix}.csv"
        )
        filepath = os.path.join(BASE_PATH, name)
        return filepath


class TextToImageBenchmark(BaseBenchmak):
    pipeline_class = AutoPipelineForText2Image

    def __init__(self, args):
        # Get number of GPUs to use
        num_gpus = getattr(args, 'num_gpus', 1)
        multi_gpu = num_gpus > 1
        
        # Set CUDA_VISIBLE_DEVICES if multiple GPUs specified
        if multi_gpu:
            device_ids = list(range(num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
            print(f"[INFO] Using {num_gpus} GPUs: {device_ids}")
        
        # Select dtype
        if args.dtype == "FP16":
            dtype = torch.float16
        elif args.dtype == "BF16":
            dtype = torch.bfloat16
        elif args.dtype == "FP32":
            dtype = torch.float32
        else:
            raise TypeError(f"Unsupported data type: {args.dtype}. "
                        f"Supported types are: BF16, FP32, FP16.")
        
        # Load pipeline with multi-GPU support for FLUX models
        if multi_gpu and "FLUX" in args.ckpt:
            print(f"[INFO] Distributing FLUX model across {num_gpus} GPUs using device_map='balanced'")
            
            # Load transformer with device_map to distribute across GPUs
            transformer = FluxTransformer2DModel.from_pretrained(
                args.ckpt,
                subfolder="transformer",
                torch_dtype=dtype,
                device_map="balanced"
            )
            
            # Load text encoder with device_map
            text_encoder_2 = T5EncoderModel.from_pretrained(
                args.ckpt,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                device_map="balanced"
            )
            
            # Load the rest of the pipeline
            pipe = FluxPipeline.from_pretrained(
                args.ckpt,
                transformer=None,
                text_encoder_2=None,
                torch_dtype=dtype
            )
            
            # Assign the distributed components
            pipe.transformer = transformer
            pipe.text_encoder_2 = text_encoder_2
            
            # Enable gradient checkpointing to reduce memory for large batches
            if hasattr(pipe.transformer, 'enable_gradient_checkpointing'):
                pipe.transformer.enable_gradient_checkpointing()
                print("[INFO] Enabled gradient checkpointing for transformer")
            
            # Move other components to first device (after CUDA_VISIBLE_DEVICES, it becomes cuda:0)
            pipe.text_encoder = pipe.text_encoder.to("cuda:0")
            pipe.vae = pipe.vae.to("cuda:0")
            
            # Enable VAE tiling for large batch sizes to reduce memory
            if args.batch_size > 4:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
                micro_batch_size = 4 if args.batch_size >= 16 else min(4, args.batch_size)
                num_micro_batches = (args.batch_size + micro_batch_size - 1) // micro_batch_size
                print(f"[INFO] Enabled VAE tiling and slicing for batch_size={args.batch_size}")
                print(f"[INFO] Will use micro-batching: {args.batch_size} images split into {num_micro_batches} batches of {micro_batch_size}")
        else:
            # Single GPU or non-FLUX models
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=dtype)
            pipe = pipe.to("cuda")

        if args.run_compile:
            if isinstance(pipe, FluxPipeline):
                pipe.transformer.to(memory_format=torch.channels_last)
                print("[INFO] Run torch compile")
                pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

            elif not isinstance(pipe, WuerstchenCombinedPipeline):
                pipe.unet.to(memory_format=torch.channels_last)
                print("[INFO] Run torch compile")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

                if hasattr(pipe, "movq") and getattr(pipe, "movq", None) is not None:
                    pipe.movq.to(memory_format=torch.channels_last)
                    pipe.movq = torch.compile(pipe.movq, mode="reduce-overhead", fullgraph=True)
            else:
                print("[INFO] Run torch compile")
                pipe.decoder = torch.compile(pipe.decoder, mode="reduce-overhead", fullgraph=True)
                pipe.vqgan = torch.compile(pipe.vqgan, mode="reduce-overhead", fullgraph=True)

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        # Use micro-batching for large batches on multi-GPU FLUX to avoid OOM
        num_gpus = getattr(args, 'num_gpus', 1)
        use_microbatching = num_gpus > 1 and "FLUX" in args.ckpt and args.batch_size > 4
        
        if use_microbatching:
            # Determine micro-batch size based on total batch size
            micro_batch_size = 4 if args.batch_size >= 16 else min(4, args.batch_size)
            num_micro_batches = (args.batch_size + micro_batch_size - 1) // micro_batch_size
            
            all_images = []
            for i in range(num_micro_batches):
                current_batch_size = min(micro_batch_size, args.batch_size - i * micro_batch_size)
                result = pipe(
                    prompt=PROMPT,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=current_batch_size,
                    height=args.resolution,
                    width=args.resolution,
                )
                all_images.extend(result.images)
        else:
            # Standard single-batch inference
            _ = pipe(
                prompt=PROMPT,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.batch_size,
                height=args.resolution,
                width=args.resolution,
            )

    def benchmark(self, args):
        flush()

        print(f"[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n")

        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        pipeline_class_name = str(self.pipe.__class__.__name__)
        flush()
        csv_dict = generate_csv_dict(
            pipeline_cls=pipeline_class_name, ckpt=args.ckpt, args=args, benchmark_info=benchmark_info
        )
        filepath = self.get_result_filepath(args)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
        flush()

class TextToImageBenchmark_multi_image(BaseBenchmak):
    pipeline_class = AutoPipelineForText2Image

    def __init__(self, args):
        # Get number of GPUs to use
        num_gpus = getattr(args, 'num_gpus', 1)
        multi_gpu = num_gpus > 1
        
        # Set CUDA_VISIBLE_DEVICES if multiple GPUs specified
        if multi_gpu:
            device_ids = list(range(num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
            print(f"[INFO] Using {num_gpus} GPUs: {device_ids}")
        
        # Select dtype
        if args.dtype == "FP16":
            dtype = torch.float16
        elif args.dtype == "BF16":
            dtype = torch.bfloat16
        elif args.dtype == "FP32":
            dtype = torch.float32
        else:
            raise TypeError(f"Unsupported data type: {args.dtype}. "
                        f"Supported types are: BF16, FP32, FP16.")
        
        # Load pipeline with multi-GPU support for FLUX models
        if multi_gpu and "FLUX" in args.ckpt:
            print(f"[INFO] Distributing FLUX model across {num_gpus} GPUs using device_map='balanced'")
            
            # Load transformer with device_map to distribute across GPUs
            transformer = FluxTransformer2DModel.from_pretrained(
                args.ckpt,
                subfolder="transformer",
                torch_dtype=dtype,
                device_map="balanced"
            )
            
            # Load text encoder with device_map
            text_encoder_2 = T5EncoderModel.from_pretrained(
                args.ckpt,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                device_map="balanced"
            )
            
            # Load the rest of the pipeline
            pipe = FluxPipeline.from_pretrained(
                args.ckpt,
                transformer=None,
                text_encoder_2=None,
                torch_dtype=dtype
            )
            
            # Assign the distributed components
            pipe.transformer = transformer
            pipe.text_encoder_2 = text_encoder_2
            
            # Enable gradient checkpointing to reduce memory for large batches
            if hasattr(pipe.transformer, 'enable_gradient_checkpointing'):
                pipe.transformer.enable_gradient_checkpointing()
                print("[INFO] Enabled gradient checkpointing for transformer")
            
            # Move other components to first device (after CUDA_VISIBLE_DEVICES, it becomes cuda:0)
            pipe.text_encoder = pipe.text_encoder.to("cuda:0")
            pipe.vae = pipe.vae.to("cuda:0")
            
            # Enable VAE tiling for large batch sizes to reduce memory
            if args.batch_size > 4:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
                micro_batch_size = 4 if args.batch_size >= 16 else min(4, args.batch_size)
                num_micro_batches = (args.batch_size + micro_batch_size - 1) // micro_batch_size
                print(f"[INFO] Enabled VAE tiling and slicing for batch_size={args.batch_size}")
                print(f"[INFO] Will use micro-batching: {args.batch_size} images split into {num_micro_batches} batches of {micro_batch_size}")
        else:
            # Single GPU or non-FLUX models
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=dtype)
            pipe = pipe.to("cuda")

        if args.run_compile:
            if isinstance(pipe, FluxPipeline):
                pipe.transformer.to(memory_format=torch.channels_last)
                print("[INFO] Run torch compile")
                pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

            elif not isinstance(pipe, WuerstchenCombinedPipeline):
                pipe.unet.to(memory_format=torch.channels_last)
                print("[INFO] Run torch compile")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

                if hasattr(pipe, "movq") and getattr(pipe, "movq", None) is not None:
                    pipe.movq.to(memory_format=torch.channels_last)
                    pipe.movq = torch.compile(pipe.movq, mode="reduce-overhead", fullgraph=True)
            else:
                print("[INFO] Run torch compile")
                pipe.decoder = torch.compile(pipe.decoder, mode="reduce-overhead", fullgraph=True)
                pipe.vqgan = torch.compile(pipe.vqgan, mode="reduce-overhead", fullgraph=True)

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        total_images = args.no_of_images
        batch_size = args.batch_size
        num_batches = (total_images + batch_size - 1) // batch_size

        all_images = []

        for _ in tqdm(range(num_batches), desc="Processing Batches"):
            current_batch_size = min(batch_size, total_images - len(all_images))
            image_output = pipe(
                prompt=PROMPT,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=current_batch_size,
                height=args.resolution,
                width=args.resolution,
            )
            images = image_output.images
            all_images.extend(images)

        print(f"Total images generated: {len(all_images)}")

    def benchmark(self, args):
        flush()

        print(f"[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n")

        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        pipeline_class_name = str(self.pipe.__class__.__name__)
        flush()
        csv_dict = generate_csv_dict(
            pipeline_cls=pipeline_class_name, ckpt=args.ckpt, args=args, benchmark_info=benchmark_info
        )
        filepath = self.get_result_filepath(args)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
        flush()

class TurboTextToImageBenchmark(TextToImageBenchmark):
    def __init__(self, args):
        super().__init__(args)

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            guidance_scale=0.0,
        )
