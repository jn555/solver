#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#include "ipm_solver.hpp"
#include "ipm_device.cuh"
#include "../utils/matrix_ops.cuh"
#include "../utils/vector_ops.cuh"

namespace cuglp
{
    IPMSolver::IPMSolver()

        : m_maxIters(1000),
          m_tolerance(1e-6)
    {
    }

    IPMSolver::~IPMSolver() = default;

    void IPMSolver::initialize(const LPProblem &problem)
    {
        m_problem = problem;
    }

    LPSolution IPMSolver::solve()
    {
        LPSolution sol;

        sol.status = SolverStatus::SUCCESS;
        sol.objectiveValue = 0.0;
        sol.solution.resize(m_problem.numVars, 0.0);

        // pass  references to the device function
        bool success = runIPMOnDevice(m_problem, m_maxIters, m_tolerance,
                                      sol.objectiveValue, sol.solution);
        if (!success)
        {
            sol.status = SolverStatus::ERROR; // or INFEASIBLE, etc.
        }
        return sol;
    }

}
