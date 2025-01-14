#ifndef CUGLP_TYPES_HPP
#define CUGLP_TYPES_HPP

#include <vector>

namespace cuglp
{

    // solver statuses
    enum class SolverStatus
    {
        SUCCESS,
        INFEASIBLE,
        UNBOUNDED,
        MAX_ITERATIONS,
        ERROR
    };

    // LP structure:
    // minimize c^T x
    // subject to Ax= b
    // x >= 0

    struct LPProblem
    {
        int numVars;        // number of variables
        int numConstraints; // number of equality constraints

        // Left ahnd size
        std::vector<double> A;
        // RHS
        std::vector<double> b;

        // Objective c, same dimensions as b
        std::vector<double> c;
    };

    struct LPSolution
    {
        SolverStatus status;
        double objectiveValue;
        std::vector<double> solution; // primal x
    };

}

#endif
