#pragma once

#include "signal.hpp"

#include <cstddef>
#include <vector>

namespace signalpipe {

struct Result {
    std::vector<Sample> spectrum;
    std::vector<float> power;
    std::size_t peak_bin = 0;
    float peak_power = 0.0f;
};

Result process_scalar(
    const std::vector<Sample>& input,
    const std::vector<float>& window
);

Result process_openmp(
    const std::vector<Sample>& input,
    const std::vector<float>& window
);

Result process_simd(
    const std::vector<Sample>& input,
    const std::vector<float>& window
);

float max_abs_error(
    const std::vector<float>& a,
    const std::vector<float>& b
);

float relative_l2_error(
    const std::vector<float>& reference,
    const std::vector<float>& actual
);

} // namespace signalpipe