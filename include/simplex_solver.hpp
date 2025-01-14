#ifndef CUGLP_SIMPLEX_SOLVER_HPP
#define CUGLP_SIMPLEX_SOLVER_HPP

#include "types.hpp"

namespace cuglp
{

    class SimplexSolver
    {
    public:
        SimplexSolver();
        ~SimplexSolver();

        void initialize(const LPProblem &problem);
        LPSolution solve();

    private:
        LPProblem m_problem;
        int m_maxIters; // max iteration cap for simplex
    };

}

#endif
