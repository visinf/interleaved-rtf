// File:   GPU.h
// Author: t-jejan
//
// Declares low-level interface functions to compute kernels that can be run on the GPU.
//
#ifndef _H_MY_GPU_H_
#define _H_MY_GPU_H_

#include <vector>
#include <memory>

namespace GPU
{
    class ConjugateGradientSolver;

    std::shared_ptr<ConjugateGradientSolver> GetSolver(const int numRows, const std::vector<int>& rowIndices,
            const int numCols, const std::vector<int>& colIndices,
            const std::vector<float>& values);

    void SolveViaConjugateGradient(std::shared_ptr<ConjugateGradientSolver>& solver, const std::vector<float>& rhs, std::vector<float>& sol, size_t maxNumIt, float residualTol);

#if 0
    void SolveViaConjugateGradient(const int numRows, const std::vector<int>& rowIndices,
                                   const int numCols, const std::vector<int>& colIndices,
                                   const std::vector<float>& values, const std::vector<float>& b,
                                   std::vector<float>& x,
                                   int maxNumIt, float residualTol);

    void SolveViaConjugateGradient(const int numRows, const std::vector<int>& rowIndices,
                                   const int numCols, const std::vector<int>& colIndices,
                                   const std::vector<double>& values, const std::vector<double>& b,
                                   std::vector<double>& x,
                                   int maxNumIt, double residualTol);
#endif
}

#endif // _H_MY_GPU_H_