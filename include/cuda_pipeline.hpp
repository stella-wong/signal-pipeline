#pragma once

#include "pipeline.hpp"

namespace signalpipe {

Result process_cuda(
    const std::vector<Sample>& input,
    const std::vector<float>& window
);

} // namespace signalpipe