#ifndef CUGLP_CUDA_UTILS_CUH
#define CUGLP_CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

namespace cuglp
{
    // Just a simple helper to check for CUDA errors.
    // Could use some improvement maybe, but it works.
    inline void checkCudaError(const char *msg)
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            // Print the error message with whatever went wrong
            std::cerr << "[CUDA Error] " << msg
                      << " : " << cudaGetErrorString(err) << std::endl;
        }
    }
}

#endif
