#ifndef CUGLP_UTILS_HPP
#define CUGLP_UTILS_HPP

#include <iostream>
#include <vector>

namespace cuglp
{

    template <typename T>
    void printMatrix(const T *A, int rows, int cols)
    {
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                std::cout << A[r * cols + c] << " ";
            }
            std::cout << "\n";
        }
    }

}

#endif
