"""Summarize Nsight SQLite exports using the CPU launch's enclosing NVTX ranges.

GPU timestamps alone cannot identify the enclosing CPU range: launches are
asynchronous. Join kernels to CUDA runtime calls by correlation ID instead.
"""

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics


def is_matmul(name):
    return bool(re.search(r"gemm|gemv|\bbmm|matmul", name, re.IGNORECASE))


def summarize(database):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        strings = dict(conn.execute("SELECT id, value FROM StringIds"))
        ranges = defaultdict(list)
        for row in conn.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL"):
            name = row["text"] or strings.get(row["textId"], "")
            ranges[name].append((row["start"], row["end"], row["globalTid"]))
        kernels = list(conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"))
        runtime = {row["correlationId"]: row for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")}

    indexes = {}
    for name, spans in ranges.items():
        spans.sort()
        indexes[name] = ([s[0] for s in spans], spans)

    def inside(name, launch):
        if name not in indexes:
            return False
        starts, spans = indexes[name]
        i = bisect_right(starts, launch["start"]) - 1
        # Autograd launches CUDA work from a worker thread while the main
        # thread's backward range is open. This harness executes steps serially,
        # so enclosing ranges in the same process also classify those launches.
        return i >= 0 and spans[i][0] <= launch["start"] <= spans[i][1] and (spans[i][2] >> 24) == (launch["globalTid"] >> 24)

    stages = ["forward", "loss", "backward", "optimizer_step", "zero_grad"]
    attention = ["computing attention scores", "computing softmax", "final matmul"]
    groups = {name: defaultdict(lambda: [0, 0]) for name in ["all", *stages, *attention]}
    step_names = sorted((name for name in ranges if re.fullmatch(r"step_\d+", name)), key=lambda name: int(name[5:]))
    per_step = {name: defaultdict(int) for name in step_names}
    unmatched = 0
    excluded = 0
    for kernel in kernels:
        launch = runtime.get(kernel["correlationId"])
        if launch is None:
            unmatched += 1
            continue
        if not inside("measure", launch):
            excluded += 1
            continue
        name = strings[kernel["demangledName"]]
        duration = kernel["end"] - kernel["start"]
        memberships = ["all", *(group for group in stages + attention if inside(group, launch))]
        for group in memberships:
            groups[group][name][0] += 1
            groups[group][name][1] += duration
        for step in step_names:
            if inside(step, launch):
                per_step[step]["kernel_count"] += 1
                per_step[step]["gpu_ns"] += duration
                break

    result = {
        "database": str(database), "steps": len(step_names),
        "captured_kernel_count": len(kernels), "unmatched_kernel_count": unmatched,
        "kernels_outside_measure": excluded, "warmup_captured": "warmup" in ranges,
        "groups": {}, "nvtx_ranges": {},
    }
    for group, stats in groups.items():
        total = sum(v[1] for v in stats.values())
        matmul = sum(v[1] for name, v in stats.items() if is_matmul(name))
        result["groups"][group] = {
            "kernel_count": sum(v[0] for v in stats.values()),
            "gpu_ms_per_step": total / 1e6 / max(1, len(step_names)),
            "matmul_ms_per_step": matmul / 1e6 / max(1, len(step_names)),
            "matmul_percent": 100 * matmul / total if total else 0,
            "kernels": [
                {"name": name, "calls": count, "calls_per_step": count / max(1, len(step_names)),
                 "gpu_ms_per_step": ns / 1e6 / max(1, len(step_names)),
                 "percent": ns / total * 100 if total else 0, "is_matmul": is_matmul(name)}
                for name, (count, ns) in sorted(stats.items(), key=lambda item: item[1][1], reverse=True)
            ],
        }
    for name in ["measure", *stages, *attention]:
        spans = ranges.get(name, [])
        result["nvtx_ranges"][name] = {
            "count": len(spans),
            "cpu_ms_per_step": sum(end - start for start, end, _ in spans) / 1e6 / max(1, len(step_names)),
        }
    result["step_wall_ms"] = [(ranges[name][0][1] - ranges[name][0][0]) / 1e6 for name in step_names]
    result["step_wall_mean_ms"] = statistics.mean(result["step_wall_ms"]) if step_names else None
    result["per_step"] = per_step
    if not step_names or not result["groups"]["all"]["kernel_count"]:
        raise ValueError("Profile contains no measured steps or correlated GPU kernels")
    if unmatched or excluded or result["warmup_captured"]:
        raise ValueError(f"Invalid capture: unmatched={unmatched}, outside_measure={excluded}, warmup={result['warmup_captured']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.database.with_suffix(".analysis.json")
    output.write_text(json.dumps(summarize(args.database), indent=2) + "\n")
    print(output)
