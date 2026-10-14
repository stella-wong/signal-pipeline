#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace signalpipe {

using Sample = std::complex<float>;

struct Config {
    std::size_t samples = 1U << 18;
    std::size_t tone_cycles = 173;
    float amplitude = 1.0f;
    float noise_stddev = 0.15f;
    std::uint32_t seed = 12345;
};

std::vector<Sample> generate_signal(const Config& config);

std::vector<float> hann_window(std::size_t size);

} // namespace signalpipe