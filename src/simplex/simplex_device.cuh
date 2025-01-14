#ifndef CUGLP_SIMPLEX_DEVICE_CUH
#define CUGLP_SIMPLEX_DEVICE_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include "../../include/utils.hpp"

namespace cuglp
{

    // Tableau on GPU in row-major. m+1 rows, n+1 cols. Includes objective row.

    // Kernel to find entering var (most neg cost)
    __global__ void findEnteringVarKernel(const double *d_tableau,
                                          int m, int n,
                                          int *d_enteringCol)
    {
        extern __shared__ double sdata[];
        int *sindex = (int *)&sdata[n];
        int idx = threadIdx.x;

        if (idx < n)
        {
            double val = d_tableau[m * (n + 1) + idx];
            sdata[idx] = val;
            sindex[idx] = idx;
        }

        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            if (idx < stride && (idx + stride) < n)
            {
                if (sdata[idx + stride] < sdata[idx])
                {
                    sdata[idx] = sdata[idx + stride];
                    sindex[idx] = sindex[idx + stride];
                }
            }
        }

        __syncthreads();
        if (idx == 0)
        {
            double bestVal = sdata[0];
            int bestIndex = sindex[0];
            if (bestVal >= 0.0)
            {
                *d_enteringCol = -1; // Done, no neg cost
            }
            else
            {
                *d_enteringCol = bestIndex;
            }
        }
    }

    // Kernel to find leaving row using ratio test
    __global__ void findLeavingRowKernel(const double *d_tableau,
                                         int m, int n,
                                         int enteringCol,
                                         int *d_leavingRow)
    {
        extern __shared__ double sdata[];
        int *sindex = (int *)&sdata[m];

        int idx = threadIdx.x;
        if (idx < m)
        {
            double colval = d_tableau[idx * (n + 1) + enteringCol];
            double rhs = d_tableau[idx * (n + 1) + n];
            if (colval > 1e-15)
            {
                double ratio = rhs / colval;
                sdata[idx] = ratio;
            }
            else
            {
                sdata[idx] = std::numeric_limits<double>::infinity();
            }
            sindex[idx] = idx;
        }
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            if (idx < stride && (idx + stride) < m)
            {
                if (sdata[idx + stride] < sdata[idx])
                {
                    sdata[idx] = sdata[idx + stride];
                    sindex[idx] = sindex[idx + stride];
                }
            }
        }

        __syncthreads();
        if (idx == 0)
        {
            if (std::isinf(sdata[0]))
            {
                *d_leavingRow = -1; // Unbounded
            }
            else
            {
                *d_leavingRow = sindex[0];
            }
        }
    }

    // Pivot kernel - kinda messy, normalize pivot row and eliminate others
    __global__ void pivotKernel(double *d_tableau,
                                int m, int n,
                                int pivotRow, int pivotCol)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < (n + 1))
        {
            double pivotVal = d_tableau[pivotRow * (n + 1) + pivotCol];
            if (fabs(pivotVal) > 1e-15)
            {
                d_tableau[pivotRow * (n + 1) + idx] /= pivotVal;
            }
        }

        __syncthreads();

        if (idx < (m + 1))
        {
            if (idx != pivotRow)
            {
                double factor = d_tableau[idx * (n + 1) + pivotCol];
                for (int j = 0; j < (n + 1); j++)
                {
                    d_tableau[idx * (n + 1) + j] -= factor * d_tableau[pivotRow * (n + 1) + j];
                }
            }
        }
    }

    // Basic Simplex method logic, some steps incomplete
    inline bool runSimplexOnDevice(const LPProblem &problem,
                                   int maxIters,
                                   double &outObj,
                                   std::vector<double> &outSolution)
    {
        const int m = problem.numConstraints;
        const int n = problem.numVars;

        size_t tableauSize = (m + 1) * (n + 1) * sizeof(double);
        double *d_tableau = nullptr;
        cudaMalloc(&d_tableau, tableauSize);

        std::vector<double> h_tableau((m + 1) * (n + 1), 0.0);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                h_tableau[i * (n + 1) + j] = problem.A[i * n + j];
            }
            h_tableau[i * (n + 1) + n] = problem.b[i];
        }

        for (int j = 0; j < n; j++)
        {
            h_tableau[m * (n + 1) + j] = problem.c[j];
        }

        cudaMemcpy(d_tableau, h_tableau.data(), tableauSize, cudaMemcpyHostToDevice);

        int enteringCol = 0;
        int leavingRow = 0;

        int *d_enteringCol = nullptr;
        int *d_leavingRow = nullptr;
        cudaMalloc(&d_enteringCol, sizeof(int));
        cudaMalloc(&d_leavingRow, sizeof(int));

        dim3 block(std::max(n, m));
        dim3 grid(1);

        bool done = false;
        int iter = 0;

        while (!done && iter < maxIters)
        {
            iter++;

            findEnteringVarKernel<<<1, block>>>(d_tableau, m, n, d_enteringCol);
            cudaDeviceSynchronize();
            int h_enteringCol;
            cudaMemcpy(&h_enteringCol, d_enteringCol, sizeof(int), cudaMemcpyDeviceToHost);

            if (h_enteringCol < 0)
            {
                done = true;
                break;
            }

            findLeavingRowKernel<<<1, block>>>(d_tableau, m, n, h_enteringCol, d_leavingRow);
            cudaDeviceSynchronize();
            int h_leavingRow;
            cudaMemcpy(&h_leavingRow, d_leavingRow, sizeof(int), cudaMemcpyDeviceToHost);

            if (h_leavingRow < 0)
            {
                cudaFree(d_tableau);
                cudaFree(d_enteringCol);
                cudaFree(d_leavingRow);
                return false;
            }

            pivotKernel<<<1, std::max(m + 1, n + 1)>>>(d_tableau, m, n, h_leavingRow, h_enteringCol);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(h_tableau.data(), d_tableau, tableauSize, cudaMemcpyDeviceToHost);

        cudaFree(d_tableau);
        cudaFree(d_enteringCol);
        cudaFree(d_leavingRow);

        if (done)
        {
            outObj = h_tableau[m * (n + 1) + n];

            outSolution.assign(n, 0.0);

            for (int j = 0; j < n; j++)
            {
                int pivotRow = -1;
                for (int i = 0; i < m; i++)
                {
                    double val = h_tableau[i * (n + 1) + j];
                    if (fabs(val - 1.0) < 1e-7)
                    {
                        pivotRow = i;
                    }
                    else if (fabs(val) > 1e-7)
                    {
                        pivotRow = -1;
                        break;
                    }
                }
                if (pivotRow >= 0)
                {
                    outSolution[j] = h_tableau[pivotRow * (n + 1) + n];
                }
            }
            return true;
        }

        return false;
    }

}

#endif
