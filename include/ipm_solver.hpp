#ifndef CUGLP_IPM_SOLVER_HPP
#define CUGLP_IPM_SOLVER_HPP

#include "types.hpp"

namespace cuglp
{

    class IPMSolver
    {
    public:
        IPMSolver();
        ~IPMSolver();

        void initialize(const LPProblem &problem);
        LPSolution solve();

    private:
        LPProblem m_problem;
        int m_maxIters; // iteration limit for ipm
        double m_tolerance;
    };

}

#endif
