#!/usr/bin/env bash

# 开启严格模式 -e 表示任意命令失败（返回非0）时脚本立即退出 -u 表示使用未定义变量时脚本立即退出
# -o pipefail 表示管道中只要有一个子命令失败，整个管道就算失败（bash默认是只看最后一个命令）
set -euo pipefail

PYTHON_CMD=python3
RUNS=10
WARMUP_STEPS=5
BENCHMARK_STEPS=10
# 没有规定 batchsize 是多少，这里就设置为 1 吧
BATCH_SIZE=1
DEVICE="auto"
RESULTS_FILE="results"

# 清空并初始化结果文件，仅记录最终结果行
: > "${RESULTS_FILE}"

# 表中的超参数设定
CONFIGS=(
  "small,768,3072,12,12"
  "medium,1024,4096,24,16"
  "large,1280,5120,36,20"
  "xl,1600,6400,48,25"
  "2.7B,2560,10240,32,32"
)

calc_stats() {
  # 使用 awk 语言计算平均值和标准差
  # awk 程序遵循 '模式 {动作}' 的形式，意思是对符合模式的行，进行规定的动作
  awk '
  # 模式为空，表示对所有的行使用
  {
  # 表示把每一行第 1 列数值存到数组 x（NR 是当前行号）
    x[NR] = $1
  # 累计求和
    sum += $1
  }
  # END 表示读完所有输入后执行
  END {
    n = NR
    if (n == 0) {
      print "nan nan"
      exit
    }
    mean = sum / n
    ss = 0
    for (i = 1; i <= n; i++) {
      d = x[i] - mean
      ss += d * d
    }
    std = sqrt(ss / n)
    printf "%.6f %.6f\n", mean, std
  }'
}

run_one_case() {
  local size="$1"
  local d_model="$2"
  local d_ff="$3"
  local num_layers="$4"
  local num_heads="$5"
  local method="$6"

  local tmp_file
  # 创建一个唯一的临时文件
  tmp_file="$(mktemp)"

  for ((i = 1; i <= RUNS; i++)); do
    echo "Running: size=${size}, method=${method}, run=${i}/${RUNS}" >&2
    "${PYTHON_CMD}" run_benchmark.py \
      --benchmark_method "${method}" \
      --warmup_steps "${WARMUP_STEPS}" \
      --benchmark_steps "${BENCHMARK_STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --d_model "${d_model}" \
      --d_ff "${d_ff}" \
      --num_layers "${num_layers}" \
      --num_heads "${num_heads}" \
      --device "${DEVICE}" >> "${tmp_file}"
  done

  read -r mean std < <(calc_stats < "${tmp_file}")
  rm -f "${tmp_file}"

  local result_line
  printf -v result_line "%-6s %-8s d_model=%-4s d_ff=%-5s layers=%-2s heads=%-2s mean=%s std=%s" \
    "${size}" "${method}" "${d_model}" "${d_ff}" "${num_layers}" "${num_heads}" "${mean}" "${std}"
  echo "${result_line}"
  echo "${result_line}" >> "${RESULTS_FILE}"
}

echo "Benchmark settings: runs=${RUNS}, warmup_steps=${WARMUP_STEPS}, benchmark_steps=${BENCHMARK_STEPS}, batch_size=${BATCH_SIZE}, device=${DEVICE}"
echo "--------------------------------------------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r size d_model d_ff num_layers num_heads <<< "${cfg}"
  run_one_case "${size}" "${d_model}" "${d_ff}" "${num_layers}" "${num_heads}" "forward"
  run_one_case "${size}" "${d_model}" "${d_ff}" "${num_layers}" "${num_heads}" "backward"
  echo "--------------------------------------------------------------------------------"
done
