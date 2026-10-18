#!/usr/bin/env bash

set -euo pipefail

# Choose which build to benchmark
if [[ "${1:-}" == "--cuda" ]]; then
    executable="./build-cuda/signal_bench"
else
    executable="./build/signal_bench"
fi

mkdir -p results

for entry in \
    "4k:4096" \
    "16k:16384" \
    "65k:65536" \
    "256k:262144" \
    "1m:1048576" \
    "4m:4194304"
do
    name="${entry%%:*}"
    samples="${entry##*:}"
    output="results/benchmark_${name}.csv"

    # Write the CSV header once, adding the run column
    echo "run,implementation,samples,median_ms,speedup,throughput_samples_s,peak_bin,peak_power,max_abs_error,relative_l2_error" > "$output"

    echo "Benchmarking ${name} samples..."

    # Perform 10 independent benchmark runs
    for run in {1..10}; do
        "$executable" "$samples" 20 |
            awk -F',' -v run="$run" \
                'NR > 1 { print run "," $0 }' >> "$output"
    done
done

echo "All benchmarks completed!"