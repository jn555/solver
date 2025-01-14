#ifndef CUGLP_IPM_DEVICE_CUH
#define CUGLP_IPM_DEVICE_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "../../include/types.hpp"

namespace cuglp
{

    // Device kernel for x = x - alpha * grad
    __global__ void ipmGradientStepKernel(double *d_x,
                                          const double *d_grad,
                                          double alpha,
                                          int n)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n)
        {
            d_x[idx] = fmax(1e-14, d_x[idx] - alpha * d_grad[idx]);
        }
    }

    // Defining a device kernel to compute the gradient of c^T x - mu * sum(log x)
    __global__ void ipmComputeGradientKernel(const double *d_x,
                                             const double *d_c,
                                             double mu,
                                             double *d_grad,
                                             int n)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n)
        {
            double val = d_c[idx] - mu / (d_x[idx] + 1e-14);
            d_grad[idx] = val;
        }
    }

    // Host function
    inline bool runIPMOnDevice(const LPProblem &problem,
                               int maxIters,
                               double tolerance,
                               double &outObj,
                               std::vector<double> &outSol)
    {
        int n = problem.numVars;
        int m = problem.numConstraints;

        double *d_x = nullptr;
        double *d_c = nullptr;
        double *d_grad = nullptr;

        cudaMalloc((void **)&d_x, n * sizeof(double));
        cudaMalloc((void **)&d_c, n * sizeof(double));
        cudaMalloc((void **)&d_grad, n * sizeof(double));

        // Host copy
        std::vector<double> h_x(n, 1.0);
        cudaMemcpy(d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, problem.c.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        double mu = 10.0;

        dim3 block(128);
        dim3 grid((n + block.x - 1) / block.x);

        for (int iter = 0; iter < maxIters; iter++)
        {
            // calculating the gradient
            ipmComputeGradientKernel<<<grid, block>>>(d_x, d_c, mu, d_grad, n);
            cudaDeviceSynchronize();

            // do a step
            double alpha = 1e-3;
            ipmGradientStepKernel<<<grid, block>>>(d_x, d_grad, alpha, n);
            cudaDeviceSynchronize();

            mu *= 0.99;

            // checking primal feasibility
            cudaMemcpy(h_x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
            double normx = 0.0;
            double cost = 0.0;
            for (int i = 0; i < n; i++)
            {
                normx += h_x[i] * h_x[i];
                cost += problem.c[i] * h_x[i];
            }
            normx = sqrt(normx);

            if (normx > 1e10)
            {
                // We say it diverged
                cudaFree(d_x);
                cudaFree(d_c);
                cudaFree(d_grad);
                return false;
            }

            if (iter % 100 == 0)
            {
                return false;
                // INCOMPLETE, not correct logic
            }
        }

        cudaMemcpy(h_x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

        // evaluatethe  objective
        double finalCost = 0.0;
        for (int i = 0; i < n; i++)
        {

            finalCost += problem.c[i] * h_x[i];
        }

        outObj = finalCost;
        outSol = h_x;

        cudaFree(d_x);

        cudaFree(d_c);
        cudaFree(d_grad);

        return true;
    }

}
#endif
