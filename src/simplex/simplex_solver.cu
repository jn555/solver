#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <cmath>

#include "simplex_solver.hpp"
#include "simplex_device.cuh"
#include "../utils/matrix_ops.cuh"
#include "../utils/vector_ops.cuh"

namespace cuglp
{

    // Ctor/dtor for solver. Nothing fancy
    SimplexSolver::SimplexSolver()
    {
        m_maxIters = 5000; // kind of arbitrary default
    }
    SimplexSolver::~SimplexSolver() = default; // nothing to cleanup yet

    void SimplexSolver::initialize(const LPProblem &problem)
    {
        m_problem = problem;
        // TODO: maybe add more checks here? currently assumes problem is well-formed
    }

    LPSolution SimplexSolver::solve()
    {
        LPSolution sol; // holds result
        sol.status = SolverStatus::SUCCESS;
        sol.objectiveValue = 0.0;
        sol.solution.resize(m_problem.numVars, 0.0); // init to zero

        // GPU solve, messy but works
        bool success = runSimplexOnDevice(m_problem, m_maxIters,
                                          sol.objectiveValue,
                                          sol.solution);
        if (!success)
        {
            sol.status = SolverStatus::ERROR; // we set status to error if something bad happens
            return sol;
        }

        // Assume success or valid result, though not perfect
        sol.status = SolverStatus::SUCCESS;
        return sol;
    }

}
