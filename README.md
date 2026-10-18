# High-Performance C++ Signal Processing Pipeline

A C++20 signal-processing project that generates complex IQ samples, applies a Hann window, performs a forward FFT, computes power, and detects the strongest frequency bin.

## Implementations

- Scalar CPU pipeline
- OpenMP-parallel windowing and power calculation
- SIMD-hinted windowing
- Optional NVIDIA CUDA and cuFFT pipeline
- CSV benchmark output
- Correctness tests against the scalar CPU implementation

FFTW performs the CPU FFT. The SIMD and OpenMP implementations use the same FFTW transform to compare the surrounding processing stages.

## Dependencies

On Ubuntu or Ubuntu under WSL:

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libfftw3-dev
```

## Build

From the project root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Benchmark

Run with the default input size and seven repetitions:

```bash
./build/signal_bench
```

Specify a sample count and number of repetitions:

```bash
./build/signal_bench [sample_count] [repetitions]
```

The program prints CSV columns for implementation, sample count, median runtime, speedup, throughput, peak bin, peak power, maximum absolute error, and relative L2 error.

## Optional CUDA build

A compatible NVIDIA GPU, NVIDIA driver, and CUDA Toolkit are required.

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON

cmake --build build-cuda -j
```

Run the CUDA-enabled benchmark:

```bash
./build-cuda/signal_bench [sample_count] [repetitions]
```

## Bash script for benchmark runs

Run 10 benchmark runs, outputting results to CSV files.

```bash
chmod +x run_benchmarks.sh
```

Without CUDA:

```bash
./run_benchmarks.sh
```

With CUDA:

```bash
./run_benchmarks.sh --cuda
```

## Benchmark interpretation

The scalar implementation is the comparison baseline. Speedup is baseline median runtime divided by the implementation's median runtime. Throughput is sample count divided by median runtime in seconds.

The CUDA implementation currently includes allocation, host-device transfers, FFT plan creation, execution, and result transfers in each timed call. It is therefore an end-to-end baseline, not a kernel-only measurement.