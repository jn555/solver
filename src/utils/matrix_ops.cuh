namespace cuglp
{

    // Kernel to multiply matrix A and vector x. Not optimized but does the job.
    __global__ void matrixVectorMultiplyKernel(const double *A,
                                               const double *x,
                                               double *y,
                                               int rows, int cols)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows)
        {
            double sum = 0.0;
            for (int c = 0; c < cols; c++)
            {
                sum += A[row * cols + c] * x[c]; // Basic dot product
            }
            y[row] = sum; // Write result to output vector
        }
    }

    // Wrapper for kernel call. Could probably use better grid/block tuning.
    inline void matrixVectorMultiply(const double *d_A,
                                     const double *d_x,
                                     double *d_y,
                                     int rows,
                                     int cols)
    {
        dim3 block(128);                           // Arbitrary block size, not tuned
        dim3 grid((rows + block.x - 1) / block.x); // Enough blocks to cover rows
        matrixVectorMultiplyKernel<<<grid, block>>>(d_A, d_x, d_y, rows, cols);
        cudaDeviceSynchronize(); // Sync to catch errors, might slow things down
    }

}

#endif
