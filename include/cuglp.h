#ifndef CUGLP_CUGLP_H
#define CUGLP_CUGLP_H

#include "types.hpp"
#include "simplex_solver.hpp"
#include "ipm_solver.hpp"
#include "utils.hpp"

namespace cuglp
{
    void initializeLibrary();
    void finalizeLibrary();
}

#endif
