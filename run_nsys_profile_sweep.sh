#!/usr/bin/env bash
set -euo pipefail

# Auto-run Nsight Systems profiling for assignment requirements:
# - model sizes: small/medium/large/xl/2.7B
# - context lengths: 128/256/512/1024
# - stages: forward / backward / train_step

# -----------------------------
# User-tunable settings
# -----------------------------
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || echo /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys)}"
PYTHON_RUNNER="${PYTHON_RUNNER:-python3}"
SCRIPT_PATH="${SCRIPT_PATH:-run_benchmark.py}"
export PYTHONPATH="$(pwd)/cs336-basics${PYTHONPATH:+:${PYTHONPATH}}"

WARMUP_STEPS="${WARMUP_STEPS:-5}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"

OUTPUT_ROOT="${OUTPUT_ROOT:-reports/nsys_profiles}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/summary.csv}"

TRACE_FLAGS="${TRACE_FLAGS:---trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite}"

# If 1, continue to next config on errors (recommended for OOM cases).
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# -----------------------------
# Fixed assignment sweep space
# -----------------------------
CONTEXT_LENGTHS=(128 256 512 1024)
METHODS=(forward backward train_step)

CONFIGS=(
  "small,768,3072,12,12"
  "medium,1024,4096,24,16"
  "large,1280,5120,36,20"
  "xl,1600,6400,48,25"
  "2.7B,2560,10240,32,32"
)

mkdir -p "${OUTPUT_ROOT}"
printf "timestamp,size,context_length,method,status,output_prefix,log_file\n" > "${SUMMARY_CSV}"

if [[ ! -x "${NSYS_BIN}" ]]; then
  echo "ERROR: nsys not found at: ${NSYS_BIN}" >&2
  echo "Tip: set NSYS_BIN, e.g. NSYS_BIN=/usr/local/cuda/bin/nsys" >&2
  exit 1
fi

run_one_case() {
  local size="$1"
  local d_model="$2"
  local d_ff="$3"
  local num_layers="$4"
  local num_heads="$5"
  local context_length="$6"
  local method="$7"

  local safe_size="${size//./p}"
  local safe_size="${safe_size// /_}"
  local prefix="${OUTPUT_ROOT}/${safe_size}_ctx${context_length}_${method}"
  local log_file="${prefix}.log"

  echo "[RUN] size=${size} ctx=${context_length} method=${method}"

  # shellcheck disable=SC2086
  if ${NSYS_BIN} profile ${TRACE_FLAGS} -o "${prefix}" \
    ${PYTHON_RUNNER} "${SCRIPT_PATH}" \
      --benchmark_method "${method}" \
      --warmup_steps "${WARMUP_STEPS}" \
      --benchmark_steps "${BENCHMARK_STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --context_length "${context_length}" \
      --d_model "${d_model}" \
      --d_ff "${d_ff}" \
      --num_layers "${num_layers}" \
      --num_heads "${num_heads}" \
      --device "${DEVICE}" \
      --annotate_attention --profile --json_output "${prefix}.timing.json" > "${log_file}" 2>&1; then
    echo "[OK ] ${prefix}.nsys-rep"
    printf "%s,%s,%s,%s,ok,%s,%s\n" \
      "$(date -Iseconds)" "${size}" "${context_length}" "${method}" "${prefix}" "${log_file}" >> "${SUMMARY_CSV}"
  else
    echo "[ERR] size=${size} ctx=${context_length} method=${method} (see ${log_file})" >&2
    printf "%s,%s,%s,%s,failed,%s,%s\n" \
      "$(date -Iseconds)" "${size}" "${context_length}" "${method}" "${prefix}" "${log_file}" >> "${SUMMARY_CSV}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
}

echo "Output root   : ${OUTPUT_ROOT}"
echo "Summary CSV   : ${SUMMARY_CSV}"
echo "NSYS binary   : ${NSYS_BIN}"
echo "Python runner : ${PYTHON_RUNNER}"
echo "Script path   : ${SCRIPT_PATH}"
echo "Sweep methods : ${METHODS[*]}"
echo "Sweep ctx     : ${CONTEXT_LENGTHS[*]}"
echo "------------------------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r size d_model d_ff num_layers num_heads <<< "${cfg}"
  for context_length in "${CONTEXT_LENGTHS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_one_case "${size}" "${d_model}" "${d_ff}" "${num_layers}" "${num_heads}" "${context_length}" "${method}"
    done
  done
done

echo "------------------------------------------------------------"
echo "Done. Summary written to: ${SUMMARY_CSV}"
echo "Example stats command:"
echo "  uv run nsys stats --report cuda_gpu_kern_sum ${OUTPUT_ROOT}/small_ctx128_forward.nsys-rep"
