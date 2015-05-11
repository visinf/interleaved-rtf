#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

#include <memory>

namespace GPU
{
    template <typename Monitor>
    void report_status(Monitor& monitor)
    {
        fprintf(stderr, "  CG: ||r||  %.2e  -->  %.2e  (%4d it)\n",
                (monitor.tolerance()/monitor.relative_tolerance()), monitor.residual_norm(), monitor.iteration_count());
    }

    class ConjugateGradientSolver
    {
    private:
        typedef cusp::device_memory memory_t;
        cusp::dia_matrix<int, float, memory_t> A;
        cusp::array1d<float, memory_t> x;
        cusp::array1d<float, memory_t> b;

    public:


        ConjugateGradientSolver(const int numRows, const std::vector<int>& rowIndices,
                                const int numCols, const std::vector<int>& colIndices,
                                const std::vector<float>& values) : x(numCols), b(numRows)
        {
            // Set up matrix in DIA format
            cusp::coo_matrix<int, float, memory_t> B(numRows, numCols, values.size());
            thrust::copy(rowIndices.begin(), rowIndices.end(), B.row_indices.begin());
            thrust::copy(colIndices.begin(), colIndices.end(), B.column_indices.begin());
            thrust::copy(values.begin(), values.end(), B.values.begin());
            A = B;
        }

        ~ConjugateGradientSolver()
        {
        }

        void Solve(const std::vector<float>& rhs, std::vector<float>& sol, size_t maxNumIt, float residualTol)
        {
            thrust::copy(rhs.begin(), rhs.end(), b.begin());
            cusp::blas::fill(x, 0);

            cusp::default_monitor<float> monitor(b, maxNumIt, residualTol, 0);

            cusp::krylov::cg(A, x, b, monitor);
            report_status(monitor);

            sol.resize(x.size());
            thrust::copy(x.begin(), x.end(), sol.begin());
        }
    };

    std::shared_ptr<ConjugateGradientSolver> GetSolver(const int numRows, const std::vector<int>& rowIndices,
            const int numCols, const std::vector<int>& colIndices,
            const std::vector<float>& values)
    {
        return std::make_shared<ConjugateGradientSolver>(numRows, rowIndices, numCols, colIndices, values);
    }

    void SolveViaConjugateGradient(std::shared_ptr<ConjugateGradientSolver>& solver,
                                   const std::vector<float>& rhs, std::vector<float>& sol,
                                   size_t maxNumIt, float residualTol)
    {
        return solver->Solve(rhs, sol, maxNumIt, residualTol);
    }
}
