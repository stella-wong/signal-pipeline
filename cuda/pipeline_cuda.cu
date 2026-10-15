#include "cuda_pipeline.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <climits>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace signalpipe {
namespace {

// error checking
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status)
        );
    }
}

// error checking
void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) +
            " failed with cuFFT error " +
            std::to_string(static_cast<int>(status))
        );
    }
}

// GPU kernel that applies the Hann window to each sample
// each GPU thread handles one sample
__global__ void window_kernel(
    const cufftComplex* input,
    const float* window,
    cufftComplex* output,
    int n
) {
    // calculate which sample this GPU thread should process
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // only process valid sample indices
    if (i < n) {
        // multiply real & imaginary parts by the window value
        output[i].x = input[i].x * window[i];
        output[i].y = input[i].y * window[i];
    }
}

// GPU kernel that calculates the power of each FFT frequency bin
__global__ void power_kernel(
    const cufftComplex* spectrum, // FFT results on the GPU
    float* power, // stores power values
    int n // number of frequency bins
) {
    // calculate which frequency bin this thread should process
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // only process valid indices
    if (i < n) {
        // get real & imaginary parts of the frequency bin
        const float re = spectrum[i].x;
        const float im = spectrum[i].y;
        // power = real^2 + imaginary^2
        power[i] = re * re + im * im;
    }
}

} // namespace

// run the signal-processing pipeline on the GPU
Result process_cuda(
    const std::vector<Sample>& input,
    const std::vector<float>& window
) {
    if (input.empty() || input.size() != window.size()) {
        throw std::invalid_argument(
            "Input and window must have the same nonzero size"
        );
    }

    if (input.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("CUDA FFT size exceeds int limit");
    }

    const int n = static_cast<int>(input.size());
    // memory needed for complex samples
    const std::size_t complex_bytes = sizeof(cufftComplex) * input.size();
    // memory for floating-point values
    const std::size_t float_bytes = sizeof(float) * input.size();

    // pointers for data stored in GPU memory
    cufftComplex* device_input = nullptr;
    cufftComplex* device_spectrum = nullptr;
    float* device_window = nullptr;
    float* device_power = nullptr;
    cufftHandle plan = 0;

    try {
        check_cuda(
            cudaMalloc(&device_input, complex_bytes),
            "Allocate input"
        );
        check_cuda(
            cudaMalloc(&device_spectrum, complex_bytes),
            "Allocate spectrum"
        );
        check_cuda(
            cudaMalloc(&device_window, float_bytes),
            "Allocate window"
        );
        check_cuda(
            cudaMalloc(&device_power, float_bytes),
            "Allocate power"
        );

        std::vector<cufftComplex> host_input(input.size());

        // convert each Sample into a cuFFT complex value
        for (std::size_t i = 0; i < input.size(); ++i) {
            host_input[i].x = input[i].real();
            host_input[i].y = input[i].imag();
        }

        // copy input signal from CPU memory to GPU memory
        check_cuda(
            cudaMemcpy(
                device_input,
                host_input.data(),
                complex_bytes,
                cudaMemcpyHostToDevice
            ),
            "Copy input"
        );

        // copy Hann window from CPU to GPU
        check_cuda(
            cudaMemcpy(
                device_window,
                window.data(),
                float_bytes,
                cudaMemcpyHostToDevice
            ),
            "Copy window"
        );

        constexpr int threads = 256;
        // blocks needed for all samples
        const int blocks = (n + threads - 1) / threads;

        // apply Hann window to the signal on the GPU
        window_kernel<<<blocks, threads>>>(
            device_input, device_window, device_spectrum, n
        );

        check_cuda(cudaGetLastError(), "Launch window kernel");

        check_cufft(
            cufftPlan1d(&plan, n, CUFFT_C2C, 1),
            "Create cuFFT plan"
        );

        // forward FFT on the windowed signal
        check_cufft(
            cufftExecC2C(
                plan,
                device_spectrum,
                device_spectrum,
                CUFFT_FORWARD
            ),
            "Execute cuFFT"
        );

        // calculate power of every frequency bin on the GPU
        power_kernel<<<blocks, threads>>>(
            device_spectrum, device_power, n
        );

        check_cuda(cudaGetLastError(), "Launch power kernel");

        std::vector<cufftComplex> host_spectrum(input.size());
        Result result;
        result.spectrum.resize(input.size());
        result.power.resize(input.size());

        // copy FFT results from GPU to CPU
        check_cuda(
            cudaMemcpy(
                host_spectrum.data(),
                device_spectrum,
                complex_bytes,
                cudaMemcpyDeviceToHost
            ),
            "Copy spectrum to host"
        );

        // copy power values from GPU to CPU
        check_cuda(
            cudaMemcpy(
                result.power.data(),
                device_power,
                float_bytes,
                cudaMemcpyDeviceToHost
            ),
            "Copy power to host"
        );

        // convert FFT results back to Sample type
        for (std::size_t i = 0; i < input.size(); ++i) {
            result.spectrum[i] = Sample(
                host_spectrum[i].x,
                host_spectrum[i].y
            );
        }

        result.peak_bin = 0;
        result.peak_power = result.power[0];

        // look for frequency bin with the highest power
        for (std::size_t i = 1; i < input.size(); ++i) {
            if (result.power[i] > result.peak_power) {
                result.peak_power = result.power[i];
                result.peak_bin = i;
            }
        }

        cufftDestroy(plan);
        cudaFree(device_input);
        cudaFree(device_spectrum);
        cudaFree(device_window);
        cudaFree(device_power);

        return result;
    } catch (...) {
        if (plan != 0) {
            cufftDestroy(plan);
        }

        if (device_input) cudaFree(device_input);
        if (device_spectrum) cudaFree(device_spectrum);
        if (device_window) cudaFree(device_window);
        if (device_power) cudaFree(device_power);

        throw;
    }
}

} // namespace signalpipe