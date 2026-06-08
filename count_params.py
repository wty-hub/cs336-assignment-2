#!/usr/bin/env python3

from __future__ import annotations

import argparse


MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute parameter count for BasicsTransformerLM-style architecture."
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=list(MODEL_SPECS.keys()),
        help="Model size preset from assignment table.",
    )
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument(
        "--weight_tying",
        action="store_true",
        help="Assume lm_head shares weights with token embedding.",
    )
    parser.add_argument(
        "--dtype_bytes",
        type=int,
        choices=[2, 4],
        default=4,
        help="Bytes per parameter for memory estimate (2=fp16/bf16, 4=fp32).",
    )
    parser.add_argument(
        "--total_only",
        action="store_true",
        help="Print total parameters only.",
    )
    return parser.parse_args()


def format_int(x: int) -> str:
    return f"{x:,}"


def main() -> None:
    args = parse_args()

    if args.size is not None:
        spec = MODEL_SPECS[args.size]
        d = spec["d_model"]
        d_ff = spec["d_ff"]
        l = spec["num_layers"]
        h = spec["num_heads"]
    else:
        required = ["d_model", "d_ff", "num_layers", "num_heads"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            missing_str = ", ".join(f"--{name}" for name in missing)
            raise ValueError(
                f"Missing required arguments: {missing_str}. "
                f"Provide all of them, or use --size from {list(MODEL_SPECS.keys())}."
            )
        d = args.d_model
        d_ff = args.d_ff
        l = args.num_layers
        h = args.num_heads

    v = args.vocab_size

    if d % h != 0:
        raise ValueError(f"d_model ({d}) must be divisible by num_heads ({h}).")

    token_embedding = v * d
    lm_head = 0 if args.weight_tying else d * v

    # Per layer:
    # Attention: q/k/v/out projection => 4 * d^2 (no bias).
    attn_per_layer = 4 * d * d
    # SwiGLU: w1(d->d_ff), w3(d->d_ff), w2(d_ff->d) => 3 * d * d_ff.
    ffn_per_layer = 3 * d * d_ff
    # Block RMSNorms: ln1 + ln2.
    norm_per_layer = 2 * d

    all_attn = l * attn_per_layer
    all_ffn = l * ffn_per_layer
    all_block_norm = l * norm_per_layer
    final_norm = d

    total = token_embedding + lm_head + all_attn + all_ffn + all_block_norm + final_norm

    if args.total_only:
        print(total)
        return

    mem_bytes = total * args.dtype_bytes
    mem_gb = mem_bytes / 1e9
    mem_gib = mem_bytes / (1024**3)

    components = [
        ("token_embedding", token_embedding),
        ("lm_head", lm_head),
        ("all_attention_proj", all_attn),
        ("all_ffn", all_ffn),
        ("all_block_norm", all_block_norm),
        ("final_norm", final_norm),
    ]

    print("Hyperparameters")
    if args.size is not None:
        print(f"  size       = {args.size}")
    print(f"  vocab_size = {v}")
    print(f"  d_model    = {d}")
    print(f"  d_ff       = {d_ff}")
    print(f"  num_layers = {l}")
    print(f"  num_heads  = {h}")
    print(f"  weight_tying = {args.weight_tying}")
    print("")

    print("Parameter breakdown")
    for name, value in components:
        ratio = (value / total * 100) if total > 0 else 0.0
        print(f"  {name:<20} {format_int(value):>15}  ({ratio:6.3f}%)")
    print(f"  {'total':<20} {format_int(total):>15}  (100.000%)")
    print("")
    print(
        f"Estimated parameter memory (@{args.dtype_bytes} bytes/param): "
        f"{mem_gb:.3f} GB ({mem_gib:.3f} GiB)"
    )


if __name__ == "__main__":
    main()
