#include "signal.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace signalpipe {

// generate a complex signal with random noise added
std::vector<Sample> generate_signal(const Config& config) {
    // check if signal has at least one sample
    if (config.samples == 0) {
        throw std::invalid_argument("Sample count must be positive");
    }

    // need number of cycles to be less than sample count
    if (config.tone_cycles >= config.samples) {
        throw std::invalid_argument(
            "tone_cycles must be less than the sample count"
        );
    }

    // vector to store all generated samples
    std::vector<Sample> output(config.samples);

    // random number generator
    std::mt19937 generator(config.seed);
    // random noise centered around zero
    std::normal_distribution<float> noise(0.0f, config.noise_stddev);

    // 2*pi for calculating the wave's angle
    constexpr float two_pi = 6.2831853071795864769f;

    // generate each sample in the signal
    for (std::size_t i = 0; i < config.samples; ++i) {
        // calculate the wave's angle at the current sample
        const float phase = two_pi * static_cast<float>(
            config.tone_cycles * i
        ) / static_cast<float>(config.samples);

        // calculate real & imaginary parts
        const float real = config.amplitude * std::cos(phase);
        const float imag = config.amplitude * std::sin(phase);

        // add random noise to both parts
        output[i] = Sample(
            real + noise(generator),
            imag + noise(generator)
        );
    }

    return output;
}

// create a Hann window to reduce edge effects before the FFT
std::vector<float> hann_window(std::size_t size) {
    std::vector<float> window(size);

    if (size == 0) {
        return window; // empty window
    }

    if (size == 1) {
        window[0] = 1.0f;
        return window;
    }

    constexpr float two_pi = 6.2831853071795864769f;

    // calculate each value in the Hann window
    for (std::size_t i = 0; i < size; ++i) {
        // values start at zero, rise toward middle, fall back to zero at the end
        window[i] = 0.5f * (
            1.0f - std::cos(
                two_pi * static_cast<float>(i) /
                static_cast<float>(size - 1)
            )
        );
    }

    return window;
}

} // namespace signalpipe