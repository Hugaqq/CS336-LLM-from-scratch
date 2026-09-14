#!/usr/bin/env bash
# Run existing A2 benchmarks sequentially; no model code is changed.
set -uo pipefail

# Edit these settings before running. All cases use the same physical GPU.
gpu_id=5
sizes=(small medium)
context_lengths=(256 512 1024)
modes=(forward_only fandb full)
timeout_seconds=180
nsys_bin=/usr/local/cuda-12.8/bin/nsys

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
cd "$project_dir" || exit 1
python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" || ! -x "$nsys_bin" ]]; then
  printf 'Missing executable: check python_bin and nsys_bin.\n' >&2
  exit 1
fi

check_gpu() {
  local gpu_status used free utilization
  gpu_status=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits) || return 1
  IFS=, read -r used free utilization <<< "$gpu_status"
  used=${used//[[:space:]]/}
  free=${free//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  if [[ ! "$used" =~ ^[0-9]+$ || ! "$free" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]]; then
    printf 'Cannot determine GPU %s availability: %s\n' "$gpu_id" "$gpu_status" >&2
    return 1
  fi
  printf 'GPU %s: used=%s MiB, free=%s MiB, utilization=%s%%. Shared use enabled; other workloads may affect timings or available memory.\n' "$gpu_id" "$used" "$free" "$utilization"
  return 0
}

check_gpu || exit 1
mkdir -p "$project_dir/results/nsight" || exit 1
results_dir=$(mktemp -d "$project_dir/results/nsight/nsys-results.XXXXXX") || exit 1
printf 'Results: %s\nGPU: %s\n' "$results_dir" "$gpu_id"
"$nsys_bin" --version > "$results_dir/nsys-version.txt"
failures=0

for size in "${sizes[@]}"; do
  for length in "${context_lengths[@]}"; do
    for mode in "${modes[@]}"; do
      check_gpu || exit 1
      tag="${size}_ctx${length}_${mode}_cuda_${gpu_id}"
      args=(-B -m cs336_systems.benchmark --compute_test --size "$size" --context_length "$length" --device "cuda:$gpu_id" "--$mode")
      printf '\nStarting %s\n' "$tag"
      timeout "$timeout_seconds" "$python_bin" "${args[@]}" 2>&1 | tee "$results_dir/${tag}_benchmark.log"
      status=$?
      printf '%s benchmark_exit=%s\n' "$tag" "$status" | tee -a "$results_dir/status.txt"
      if (( status != 0 )); then
        failures=$((failures + 1))
        continue
      fi

      check_gpu || exit 1
      env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN timeout "$timeout_seconds" "$nsys_bin" profile \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=none \
        --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
        -o "$results_dir/$tag" "$python_bin" "${args[@]}" \
        2>&1 | tee "$results_dir/${tag}_nsys.log"
      status=$?
      printf '%s profile_exit=%s\n' "$tag" "$status" | tee -a "$results_dir/status.txt"
      if (( status != 0 )); then
        failures=$((failures + 1))
      fi
    done
  done
done

printf '\nFinished. Failed runs: %s\nResults: %s\n' "$failures" "$results_dir"
(( failures == 0 ))
