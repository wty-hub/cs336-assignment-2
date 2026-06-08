# run_benchmark.py

import argparse
import math
import timeit

import cs336_basics.model as model_module
import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
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


def run_step(model, batch, benchmark_method, optimizer=None):
    if benchmark_method == "forward":
        with nvtx.range("forward"):
            model(batch)
    elif benchmark_method == "backward":
        with nvtx.range("forward"):
            output = model(batch)
        with nvtx.range("backward"):
            output.mean().backward()
        model.zero_grad(set_to_none=True)
    elif benchmark_method == "train_step":
        if optimizer is None:
            raise ValueError("optimizer is required for train_step")
        with nvtx.range("forward"):
            output = model(batch)
        with nvtx.range("backward"):
            output.mean().backward()
        with nvtx.range("optimizer_step"):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        raise ValueError(f"Unknown benchmark_method: {benchmark_method}")


if __name__ == "__main__":
    args = parse_args()
    device = resolve_device(args.device)

    if args.annotate_attention:
        model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = get_model(args).to(device)
    optimizer = None
    if args.benchmark_method == "train_step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 预热
    maybe_sync(device)
    warmup_start_time = timeit.default_timer()
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            batch = generate_random_batch(args.batch_size, args.context_length, device)
            run_step(model, batch, args.benchmark_method, optimizer)
    maybe_sync(device)

    # 开始基准测试
    maybe_sync(device)
    batches = [generate_random_batch(args.batch_size, args.context_length, device) for _ in range(args.benchmark_steps)]
    benchmark_start_time = timeit.default_timer()
    with nvtx.range("measure"):
        for batch in batches:
            run_step(model, batch, args.benchmark_method, optimizer)
    maybe_sync(device)
    benchmark_end_time = timeit.default_timer()

    if args.verbose:
        print(
            f"Benchmark completed in {benchmark_end_time - benchmark_start_time:.2f} seconds for {args.benchmark_steps} steps,"
            f"average {((benchmark_end_time - benchmark_start_time) / args.benchmark_steps):.2f} seconds per step"
        )
    else:
        print(f'{benchmark_end_time - benchmark_start_time}')