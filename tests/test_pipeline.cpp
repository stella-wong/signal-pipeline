#include "pipeline.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

// helper to check if condition is true
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// test that a generated tone has the expected FFT peak
// test that the OpenMP and SIMD results match the scalar result
void test_tone_peak() {
    // configure test signal
    signalpipe::Config config;
    config.samples = 4096;
    config.tone_cycles = 37;
    config.noise_stddev = 0.0f;

    // generate test signal and Hann window
    const auto input = signalpipe::generate_signal(config);
    const auto window = signalpipe::hann_window(config.samples);

    // process signal using 3 CPU implementations
    const auto scalar =
        signalpipe::process_scalar(input, window);
    const auto openmp =
        signalpipe::process_openmp(input, window);
    const auto simd =
        signalpipe::process_simd(input, window);

    const std::size_t n = config.samples;

    // check that the strongest FFT bin is the expected one
    // peak can be either 37 (positive frequency) or
    // 4096 - 37 (corresponding negative frequency bin)
    require(
        scalar.peak_bin == config.tone_cycles ||
        scalar.peak_bin == n - config.tone_cycles,
        "Unexpected peak frequency bin"
    );

    // compare OpenMP power values with scalar reference
    require(
        signalpipe::max_abs_error(
            scalar.power, openmp.power
        ) < 1e-2f,
        "OpenMP result differs from scalar"
    );

    // compare SIMD with scalar
    require(
        signalpipe::max_abs_error(
            scalar.power, simd.power
        ) < 1e-2f,
        "SIMD result differs from scalar"
    );
}

// test with input signal of only zeros
void test_zero_input() {
    constexpr std::size_t n = 128;

    std::vector<signalpipe::Sample> input(
        n, signalpipe::Sample(0.0f, 0.0f)
    );
    // window of all ones, so the input is unchanged
    std::vector<float> window(n, 1.0f);

    const auto result =
        signalpipe::process_scalar(input, window);

    require(result.peak_power == 0.0f, "Zero input has nonzero power");
    require(result.peak_bin == 0, "Zero input has unexpected peak bin");
}

} // namespace

int main() {
    try {
        test_tone_peak();
        test_zero_input();

        std::cout << "All tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}