# run_benchmark.py

import argparse
import json
import math
from pathlib import Path
import statistics
import timeit

import cs336_basics.model as model_module
import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--json_output", type=Path)
    parser.add_argument("--profile", action="store_true", help="Capture only measured steps with the CUDA profiler API")
    parser.add_argument("--cpu_threads", type=int, default=4)
    # 预热步数
    parser.add_argument("--warmup_steps", type=int, default=10)
    # 规定的参数：基准测试方法
    parser.add_argument(
        "--benchmark_method",
        type=str,
        choices=["forward", "backward", "train_step"],
        default="forward",
    )

    parser.add_argument("--benchmark_steps", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=1024)

    # 模型参数，只使用表中的参数
    ## d_model 是 Transformer 最后一维的维度
    parser.add_argument("--d_model", type=int, required=True)
    ## d_ff 是 Feed Forward 层的内部维度
    parser.add_argument("--d_ff", type=int, required=True)
    ## num_layers 是 Transformer 的层数
    parser.add_argument("--num_layers", type=int, required=True)
    ## num_heads 是 Multi-Head Attention 的头数
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='device: cuda、cuda:0、cpu，或 auto（有 GPU 则用 cuda，否则 cpu）',
    )

    # 是否使用详细输出
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument(
        "--annotate_attention",
        action="store_true",
        help="Replace scaled_dot_product_attention with an NVTX-annotated version for nsys profiling",
    )

    return parser.parse_args()


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_model(args):
    return BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )


def generate_random_batch(batch_size, context_length, device: torch.device):
    return torch.randint(0, 10000, (batch_size, context_length), device=device)


def run_step(model, batch, benchmark_method, optimizer=None, targets=None):
    if benchmark_method == "forward":
        with torch.no_grad(), nvtx.range("forward"):
            model(batch)
    elif benchmark_method in {"backward", "train_step"}:
        if targets is None:
            raise ValueError("targets are required for cross-entropy loss")
        if benchmark_method == "train_step" and optimizer is None:
            raise ValueError("optimizer is required for train_step")
        with nvtx.range("forward"):
            output = model(batch)
        with nvtx.range("loss"):
            loss = cross_entropy(output, targets)
        with nvtx.range("backward"):
            loss.backward()
        if benchmark_method == "train_step":
            with nvtx.range("optimizer_step"):
                optimizer.step()
        with nvtx.range("zero_grad"):
            model.zero_grad(set_to_none=True)
    else:
        raise ValueError(f"Unknown benchmark_method: {benchmark_method}")


if __name__ == "__main__":
    args = parse_args()
    if args.warmup_steps < 0 or args.benchmark_steps < 1:
        raise ValueError("warmup_steps must be nonnegative and benchmark_steps must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    if args.annotate_attention:
        model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # Initialize directly on the requested device, including large configurations
    # that may exceed GPU memory. Each sweep case runs in a separate process.
    with torch.device(device):
        model = get_model(args)
    model.train(args.benchmark_method != "forward")
    optimizer = None
    if args.benchmark_method == "train_step":
        optimizer = AdamW(model.parameters(), lr=1e-4)

    tokens = generate_random_batch(args.batch_size, args.context_length + 1, device)
    batch, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    # 预热
    maybe_sync(device)
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            run_step(model, batch, args.benchmark_method, optimizer, targets)
            maybe_sync(device)
    maybe_sync(device)

    # 开始基准测试
    if args.profile:
        if device.type != "cuda":
            raise ValueError("--profile requires CUDA")
        torch.cuda.cudart().cudaProfilerStart()
    step_seconds = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    benchmark_start_time = timeit.default_timer()
    with nvtx.range("measure"):
        for i in range(args.benchmark_steps):
            step_start = timeit.default_timer()
            with nvtx.range(f"step_{i}"):
                run_step(model, batch, args.benchmark_method, optimizer, targets)
                maybe_sync(device)
            step_seconds.append(timeit.default_timer() - step_start)
    maybe_sync(device)
    benchmark_end_time = timeit.default_timer()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "optimizer": "cs336_basics.optimizer.AdamW" if optimizer is not None else None,
            "loss": "next-token cross entropy" if args.benchmark_method != "forward" else None,
            "step_seconds": step_seconds,
            "mean_seconds": statistics.mean(step_seconds),
            "std_seconds": statistics.pstdev(step_seconds),
            "total_seconds": benchmark_end_time - benchmark_start_time,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")

    if args.verbose:
        print(
            f"Benchmark completed in {benchmark_end_time - benchmark_start_time:.2f} seconds for {args.benchmark_steps} steps,"
            f"average {((benchmark_end_time - benchmark_start_time) / args.benchmark_steps):.2f} seconds per step"
        )
    else:
        print(f'{benchmark_end_time - benchmark_start_time}')
