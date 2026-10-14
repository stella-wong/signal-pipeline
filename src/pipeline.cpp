#include "pipeline.hpp"

#include <fftw3.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef SIGNAL_HAS_OPENMP
#include <omp.h>
#endif

namespace signalpipe {
namespace {

// check that the input signal and window have valid, matching sizes
void validate(
    const std::vector<Sample>& input,
    const std::vector<float>& window
) {
    if (input.empty()) {
        throw std::invalid_argument("Input must not be empty");
    }

    if (input.size() != window.size()) {
        throw std::invalid_argument(
            "Input and window sizes must match"
        );
    }
}

// apply window to input signal, perform FFT, return frequency spectrum
Result calculate_fft(
    const std::vector<Sample>& input,
    const std::vector<float>& window,
    bool parallel_window,
    bool simd_window
) {
    validate(input, window);

    const std::size_t n = input.size();

    if (n > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("Input exceeds FFTW size limit");
    }

    fftwf_complex* buffer =
        static_cast<fftwf_complex*>(fftwf_malloc(
            sizeof(fftwf_complex) * n
        ));

    if (buffer == nullptr) {
        throw std::bad_alloc();
    }

    // multiply each complex's parts by its window value
    for (std::size_t i = 0; i < n; ++i) {
        buffer[i][0] = input[i].real() * window[i];
        buffer[i][1] = input[i].imag() * window[i];
    }

#ifdef SIGNAL_HAS_OPENMP
    // parallelize windowing across multiple CPU threads
    if (parallel_window) {
        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            buffer[i][0] = input[i].real() * window[i];
            buffer[i][1] = input[i].imag() * window[i];
        }
    }
#endif

    // request SIMD vectorization to process multiple samples at once
    if (simd_window) {
        #pragma omp simd
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            buffer[i][0] = input[i].real() * window[i];
            buffer[i][1] = input[i].imag() * window[i];
        }
    }

    fftwf_plan plan = fftwf_plan_dft_1d(
        static_cast<int>(n),
        buffer,
        buffer,
        FFTW_FORWARD,
        FFTW_ESTIMATE
    );

    if (plan == nullptr) {
        fftwf_free(buffer);
        throw std::runtime_error("FFTW plan creation failed");
    }

    fftwf_execute(plan);

    Result result;
    result.spectrum.resize(n);
    result.power.resize(n);

    // copy the FFT output into the result
    for (std::size_t i = 0; i < n; ++i) {
        result.spectrum[i] = Sample(buffer[i][0], buffer[i][1]);
    }

    fftwf_destroy_plan(plan);
    fftwf_free(buffer);

    return result;
}

// calculate the power of each frequency bin, find the bin with the highest power
void calculate_power(Result& result, bool parallel) {
    const std::size_t n = result.spectrum.size();

#ifdef SIGNAL_HAS_OPENMP
    // calculate power for multiple bins in parallel
    if (parallel) {
        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const float re = result.spectrum[i].real();
            const float im = result.spectrum[i].imag();
            result.power[i] = re * re + im * im;
        }
    } else
#endif
    {
        // calculate power one bin at a time
        for (std::size_t i = 0; i < n; ++i) {
            const float re = result.spectrum[i].real();
            const float im = result.spectrum[i].imag();
            result.power[i] = re * re + im * im;
        }
    }

    result.peak_bin = 0;
    result.peak_power = result.power[0];

    // look for highest power
    for (std::size_t i = 1; i < n; ++i) {
        if (result.power[i] > result.peak_power) {
            result.peak_power = result.power[i];
            result.peak_bin = i;
        }
    }
}

} // namespace

// scalar implementation for windowing, FFT, power calculation
Result process_scalar(
    const std::vector<Sample>& input,
    const std::vector<float>& window
) {
    Result result = calculate_fft(input, window, false, false);
    calculate_power(result, false);
    return result;
}

// OpenMP implementation with CPU multithreading for windowing and power calculation
Result process_openmp(
    const std::vector<Sample>& input,
    const std::vector<float>& window
) {
    Result result = calculate_fft(input, window, true, false);
    calculate_power(result, true);
    return result;
}

// SIMD implementation requests vectorization for windowing (power calculation is scalar)
Result process_simd(
    const std::vector<Sample>& input,
    const std::vector<float>& window
) {
    Result result = calculate_fft(input, window, false, true);
    calculate_power(result, false);
    return result;
}

// return largest absolute difference between two power arrays
float max_abs_error(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes differ");
    }

    float error = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i) {
        error = std::max(error, std::abs(a[i] - b[i]));
    }

    return error;
}

// compare the results with the reference and measure the error
float relative_l2_error(
    const std::vector<float>& reference,
    const std::vector<float>& actual
) {
    if (reference.size() != actual.size()) {
        throw std::invalid_argument("Vector sizes differ");
    }

    double error_squared = 0.0;
    double reference_squared = 0.0;

    // sum the squared differences and squared reference values
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double difference =
            static_cast<double>(actual[i]) - reference[i];

        error_squared += difference * difference;
        reference_squared +=
            static_cast<double>(reference[i]) * reference[i];
    }

    // handle division by zero
    if (reference_squared == 0.0) {
        return error_squared == 0.0
            ? 0.0f
            : static_cast<float>(std::sqrt(error_squared));
    }

    // return relative L2 error; lower values mean closer results
    return static_cast<float>(
        std::sqrt(error_squared / reference_squared)
    );
}

} // namespace signalpipe