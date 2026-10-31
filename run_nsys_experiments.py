"""Run the assignment's full Nsight sweep sequentially, preserving raw evidence."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from analyze_nsys import summarize


CONFIGS = {
    "small": (768, 3072, 12, 12), "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20), "xl": (1600, 6400, 48, 25),
    "2.7B": (2560, 10240, 32, 32),
}


def invoke(command, log_path, env):
    with log_path.open("w") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    log = log_path.read_text(errors="replace")
    if "OutOfMemoryError" in log or "CUDA out of memory" in log:
        return "oom", code
    return ("ok" if code == 0 else "failed"), code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", default=shutil.which("nsys") or "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--methods", nargs="+", default=["forward", "backward", "train_step"], choices=["forward", "backward", "train_step"])
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "cs336-basics") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    import torch
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version,
        "platform": platform.platform(),
        "nsys": subprocess.check_output([args.nsys, "--version"], text=True).strip(),
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "batch_size": 4, "dtype": "float32", "vocab_size": 10000,
        "warmup_steps": args.warmup_steps, "benchmark_steps": args.benchmark_steps,
        "inference_grad_enabled": False, "loss": "next-token cross entropy",
        "optimizer": "cs336_basics.optimizer.AdamW (provided assignment-1 implementation)",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Persist the precise scripts used even when the working tree later changes.
    for script in ["run_benchmark.py", "run_nsys_experiments.py", "analyze_nsys.py"]:
        shutil.copy2(root / script, args.output / (script + ".snapshot"))
    rows = []
    for size in args.sizes:
        dm, dff, nl, nh = CONFIGS[size]
        for context in args.contexts:
            common = [sys.executable, str(root / "run_benchmark.py"), "--d_model", str(dm),
                      "--d_ff", str(dff), "--num_layers", str(nl), "--num_heads", str(nh),
                      "--context_length", str(context), "--batch_size", "4", "--device", "cuda",
                      "--warmup_steps", str(args.warmup_steps), "--benchmark_steps", str(args.benchmark_steps),
                      "--annotate_attention"]
            for method in args.methods:
                stem = args.output / f"{size.replace('.', 'p')}_ctx{context}_{method}"
                print(f"RUN {size} ctx={context} {method}", flush=True)
                baseline_status = "not_run"
                if method == "forward":
                    baseline_status, _ = invoke(common + ["--benchmark_method", method, "--json_output", str(stem) + ".baseline.json"], Path(str(stem) + ".baseline.log"), env)
                command = [args.nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                           "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--export=sqlite",
                           "--force-overwrite=true", "-o", str(stem), *common,
                           "--benchmark_method", method, "--profile", "--json_output", str(stem) + ".timing.json"]
                status, code = invoke(command, Path(str(stem) + ".log"), env)
                error = ""
                if status == "ok":
                    try:
                        analysis = summarize(Path(str(stem) + ".sqlite"))
                        if analysis["steps"] != args.benchmark_steps:
                            raise ValueError("Wrong measured step count")
                        if not Path(str(stem) + ".timing.json").exists():
                            raise ValueError("Missing application timing output")
                        Path(str(stem) + ".analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
                    except Exception as exc:
                        status, error = "invalid_profile", str(exc)
                row = {"size": size, "context_length": context, "method": method, "status": status,
                       "exit_code": code, "baseline_status": baseline_status, "output_prefix": str(stem), "error": error}
                rows.append(row)
                with (args.output / "summary.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"RESULT {size} ctx={context} {method}: {status} baseline={baseline_status} {error}", flush=True)


if __name__ == "__main__":
    main()
