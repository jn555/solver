#ifndef CUGLP_VECTOR_OPS_CUH
#define CUGLP_VECTOR_OPS_CUH

#include <cuda_runtime.h>

namespace cuglp
{

    // Kernel to scale a vector. Just multiplies each element by alpha.
    __global__ void scaleVectorKernel(double *vec, double alpha, int n)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n)
        {
            vec[idx] *= alpha; // Pretty straightforward scaling
        }
    }

    // Wrapper for the scaling kernel. Simple, not much error checking.
    inline void scaleVector(double *d_vec, double alpha, int n)
    {
        dim3 block(128);                        // Fixed block size, might not be optimal
        dim3 grid((n + block.x - 1) / block.x); // Grid size calculated to cover all elements
        scaleVectorKernel<<<grid, block>>>(d_vec, alpha, n);
        cudaDeviceSynchronize(); // Sync to ensure kernel finishes
    }

}

#endif
