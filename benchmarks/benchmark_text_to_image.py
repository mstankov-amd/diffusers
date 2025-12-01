import argparse
import sys


sys.path.append(".")
from base_classes import TextToImageBenchmark, TurboTextToImageBenchmark, TextToImageBenchmark_multi_image  # noqa: E402


ALL_T2I_CKPTS = [
    "Lykon/DreamShaper",
    "segmind/SSD-1B",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "kandinsky-community/kandinsky-2-2-decoder",
    "warp-ai/wuerstchen",
    "stabilityai/sdxl-turbo",
    "etri-vilab/koala-1b",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="Lykon/DreamShaper",
        choices=ALL_T2I_CKPTS,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="FP16",
        choices=("FP16", "FP32", "BF16"),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    parser.add_argument("--no_of_images", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--device-ids", 
        type=str, 
        default="0",
        help="Comma-separated GPU device IDs (e.g., '0' for single GPU, '0,1,2' for multi-GPU with device_map)"
    )
    args = parser.parse_args()

    benchmark_cls = None
    if "turbo" in args.ckpt:
        benchmark_cls = TurboTextToImageBenchmark
    elif args.no_of_images > 1:
        benchmark_cls = TextToImageBenchmark_multi_image
    else:
        benchmark_cls = TextToImageBenchmark

    benchmark_pipe = benchmark_cls(args)
    benchmark_pipe.benchmark(args)
