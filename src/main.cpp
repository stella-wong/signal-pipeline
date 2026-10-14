#include "pipeline.hpp"

#ifdef SIGNAL_ENABLE_CUDA
#include "cuda_pipeline.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// stores the performance & accuracy results for each implementation
struct Benchmark {
    std::string name;
    double median_ms;
    double speedup;
    double throughput;
    std::size_t peak_bin;
    float peak_power;
    float max_error;
    float relative_error;
};

// measures how long one execution of a function takes (in milliseconds)
template <typename Function>
double time_ms(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto end = Clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

// run function repeatedly, return median time
template <typename Function>
double benchmark_median(
    Function&& function,
    int warmups,
    int repetitions
) {
    for (int i = 0; i < warmups; ++i) {
        function();
    }

    std::vector<double> times;
    times.reserve(repetitions);

    for (int i = 0; i < repetitions; ++i) {
        times.push_back(time_ms(function));
    }

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    try {
        // set default input size & number of benchmark repetitions
        std::size_t sample_count = 1U << 18;
        int repetitions = 7;

        // allow user to override the defaults using command-line arguments
        if (argc > 1) {
            sample_count = std::stoull(argv[1]);
        }

        if (argc > 2) {
            repetitions = std::stoi(argv[2]);
        }

        // argument restrictions
        if (sample_count < 2 || sample_count > (1U << 26)) {
            throw std::invalid_argument(
                "Sample count must be between 2 and 2^26"
            );
        }

        if (repetitions < 1 || repetitions > 1000) {
            throw std::invalid_argument(
                "Repetitions must be between 1 and 1000"
            );
        }

        // configure & generate the input signal and its Hann window
        signalpipe::Config config;
        config.samples = sample_count;
        config.tone_cycles = 173;

        const auto input = signalpipe::generate_signal(config);
        const auto window = signalpipe::hann_window(sample_count);

        // run scalar implementation to provide a reference for accuracy checks
        signalpipe::Result reference =
            signalpipe::process_scalar(input, window);

        std::vector<Benchmark> results;

        // benchmark the scalar implementation to establish baseline time
        const double baseline_ms = benchmark_median(
            [&] {
                reference = signalpipe::process_scalar(input, window);
            },
            2,
            repetitions
        );

        // save performance metrics, compare each result with the scalar reference
        auto record = [&](const std::string& name,
                          const signalpipe::Result& result,
                          double median_ms) {
            results.push_back({
                name,
                median_ms,
                baseline_ms / median_ms,
                static_cast<double>(sample_count) /
                    (median_ms / 1000.0),
                result.peak_bin,
                result.peak_power,
                signalpipe::max_abs_error(
                    reference.power, result.power
                ),
                signalpipe::relative_l2_error(
                    reference.power, result.power
                )
            });
        };

        // record the scalar baseline results
        auto scalar_result = signalpipe::process_scalar(input, window);
        record("scalar", scalar_result, baseline_ms);

        // benchmark the CPU-parallel OpenMP implementation
        signalpipe::Result openmp_result;
        const double openmp_ms = benchmark_median(
            [&] {
                openmp_result =
                    signalpipe::process_openmp(input, window);
            },
            2,
            repetitions
        );
        record("openmp", openmp_result, openmp_ms);

        // benchmark the SIMD implementation, which uses CPU vector instructions
        signalpipe::Result simd_result;
        const double simd_ms = benchmark_median(
            [&] {
                simd_result =
                    signalpipe::process_simd(input, window);
            },
            2,
            repetitions
        );
        record("simd", simd_result, simd_ms);

#ifdef SIGNAL_ENABLE_CUDA
        // if CUDA is enabled, benchmark the GPU implementation
        signalpipe::Result cuda_result;
        const double cuda_ms = benchmark_median(
            [&] {
                cuda_result =
                    signalpipe::process_cuda(input, window);
            },
            2,
            repetitions
        );
        record("cuda", cuda_result, cuda_ms);
#endif

        // print results as CSV
        std::cout
            << "implementation,samples,median_ms,speedup,"
            << "throughput_samples_s,peak_bin,peak_power,"
            << "max_abs_error,relative_l2_error\n";

        std::cout << std::setprecision(9);

        for (const auto& result : results) {
            std::cout
                << result.name << ','
                << sample_count << ','
                << result.median_ms << ','
                << result.speedup << ','
                << result.throughput << ','
                << result.peak_bin << ','
                << result.peak_power << ','
                << result.max_error << ','
                << result.relative_error << '\n';
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}