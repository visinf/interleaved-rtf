// File:   Minimization.h
// Author: t-jejan
//
// Implements various routines for constrained convex optimization problems for which efficient
// projections onto the feasible set are available.
// Unconstrained problems can also be handled, either as a special case of a constrained problem
// for which projections amount to identity, or using one of the specialized methods for
// unconstrained optimization, such as L-BFGS.
//
// Currently, we provide the following methods:
//
// - SPGMinimize()
//     Implements the spectral projected gradient method with Barzilai-Borwein scaling and
//     a monotone line search.
// - PQNMinimize()
//     Implements the projected quasi-Newton method by Mark Schmidt, as described in
//     http://jmlr.csail.mit.edu/proceedings/papers/v5/schmidt09a/schmidt09a.pdf.
// - RestartingLBFGSMinimize()
//     A heuristic method that uses the L-BFGS approximation to the inverse Hessian to try and
//     determine a descent direction. Note that in a constrained setting, we are not actually
//     guarantee to find direction of descent this way. Hence, if line search fails, we reset
//     the approximation of the inverse Hessian to the identity matrix, such that we take a
//     plain projected gradient step, and then starting building up the approximation again.
//     For some constraint sets, this works surprisingly well.
// - LBFGSMinimize()
//     The classic LBFGS method for unconstrained problems.
// - CGSolve()
//     Solves a sparse linear system Ax = b for A positive definite using conjugate gradient.
//     Note that this is equivalent to minimizing the unconstrained quadratic 1/2 x^TAx - x^Tb.
//
#ifndef _H_MINIMIZATION_H_
#define _H_MINIMIZATION_H_

#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <random>
#include <iostream>

#include <Eigen/Dense>

namespace Minimization
{
    template<typename TValue>
    class Types
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
    };

    template<typename TValue>
    class ProjectableProblem
    {
    public:
        typedef typename Types<TValue>::TVector TVector;

        virtual ~ProjectableProblem() { }

        virtual TValue Eval(const TVector& x, TVector& g) = 0;

        virtual unsigned Dimensions() const = 0;

        virtual void ProvideStartingPoint(TVector& x0) const = 0;

        virtual TVector Project(const TVector& x) const = 0;

        virtual bool IsFeasible(const TVector& x) const = 0;

        virtual TValue Norm(const TVector& g) const
        {
            return g.norm();
        }

        virtual void Report(const char* fmt, ...) const
        {
            va_list args;
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
        }
    };

    template<typename TValue>
    class UnconstrainedProblem : public ProjectableProblem<TValue>
    {
    public:
        typedef ProjectableProblem<TValue> Base;

        typename Base::TVector Project(const typename Base::TVector& x) const
        {
            return x;
        }

        bool IsFeasible(const typename Base::TVector& x) const
        {
            return true;
        }
    };

    // Port of senowozi's CheckDerivative tool to our function minimization interface
    template<typename TValue>
    bool CheckDerivative(ProjectableProblem<TValue>& prob,
                         TValue x_range, unsigned int test_count, TValue dim_eps, TValue grad_tol)
    {
        assert(dim_eps > 0.0);
        assert(grad_tol > 0.0);
        typedef typename ProjectableProblem<TValue>::TVector TVector;
        // Random number generation, for random perturbations
        std::mt19937 rgen; //(static_cast<unsigned>(std::time(0)) + 1);
        std::uniform_real_distribution<TValue> rdestu;	// range [0,1]
        // Random number generation, for random dimensions
        unsigned int dim = prob.Dimensions();
        std::mt19937 rgen2; //(static_cast<unsigned>(std::time(0)) + 2);
        std::uniform_int_distribution<unsigned int> rdestd(0, dim - 1);
        // Get base
        TVector x0(dim);
        prob.ProvideStartingPoint(x0);
        TVector xtest(dim);
        TVector grad(dim);
        TVector grad_d(dim);	// dummy

        for(unsigned int test_id = 0; test_id < test_count; ++test_id)
        {
            xtest = x0;

            for(unsigned int d = 0; d < dim; ++d)
                xtest[d] += 2.0 * x_range * rdestu(rgen) - x_range;

            //xtest = prob.Project(xtest); // ensure the point is feasible
            // Get exact derivative
            TValue xtest_fval = prob.Eval(xtest, grad);
            // Compute first-order finite difference approximation
            unsigned int test_dim = rdestd(rgen);
            xtest[test_dim] += dim_eps;
            TValue xtest_d_fval = prob.Eval(xtest, grad_d);
            TValue deriv_fd = (xtest_d_fval - xtest_fval) / dim_eps;
            std::cerr << "testval: " << xtest_fval << std::endl;
            std::cerr << "testdval: " << xtest_d_fval << std::endl;

            // Check accuracy
            if(std::abs(deriv_fd - grad[test_dim]) > grad_tol)
            {
                std::ios_base::fmtflags original_format = std::cout.flags();
                std::streamsize original_prec = std::cout.precision();
                std::cout << std::endl;
                std::cout << "### DERIVATIVE CHECKER WARNING" << std::endl;
                std::cout << "### during test " << (test_id + 1) << " a violation "
                          << "in gradient computation was found:" << std::endl;
                std::cout << std::setprecision(6)
                          << std::setiosflags(std::ios::scientific);
                std::cout << "### dim " << test_dim << ", exact " << grad[test_dim]
                          << ", finite-diff " << deriv_fd
                          << ", absdiff " << fabs(deriv_fd - grad[test_dim])
                          << std::endl;
                std::cout << std::endl;
                std::cout.precision(original_prec);
                std::cout.flags(original_format);
                //Sleep(1000 * 60 * 60 * 10);
                //return (false);
            } else {
                std::cout << "### dim " << test_dim << " passed!" << std::endl;
                //Sleep(2000);
            }
        }

        return (true);
    }

    template<typename VectorType, int k>
    void
    ProjectOntoUnitSimplex(Eigen::VectorBlock<VectorType, k> v)
    {
        typedef typename VectorType::Scalar TValue;
        Eigen::Matrix<TValue, k, 1> mu = v;
        std::sort(mu.data(), mu.data()+k); // sorted in ascending order
        size_t j  = 1;
        TValue s  = 0.0;
        TValue r  = 0.0;
        TValue sr = 0.0;

        while( j <= k ) {
            const auto m = mu(k-j);
            s += m;
            if( (m - (1.0/j)*(s - 1.0)) > 0.0) {
                r  = j;
                sr = s;
            }
            j += 1;
        }
        const auto theta = (1.0/r)*(sr-1.0);
        j = 0;
        while( j < k ) {
            v(j) = std::max(0.0, v(j) - theta);
            j += 1;
        }
    }


    template<typename TValue>
    TValue ProjectedGradientNorm(const ProjectableProblem<TValue>& problem,
                                 const typename Types<TValue>::TVector& x,
                                 const typename Types<TValue>::TVector& g)
    {
        return problem.Norm((problem.Project(x - g) - x));
    }

    template<typename TValue>
    TValue SPGComputeDirection(const ProjectableProblem<TValue>& problem,
                               const typename Types<TValue>::TVector& x,
                               const typename Types<TValue>::TVector& g,
                               TValue alpha,
                               typename Types<TValue>::TVector& d)
    {
        d = problem.Project(x - alpha * g) - x;
        return d.dot(g);
    }

    template<typename TValue>
    TValue SPGComputeStepSize(const ProjectableProblem<TValue>& problem,
                              const typename Types<TValue>::TVector& newx, const typename Types<TValue>::TVector& oldx,
                              const typename Types<TValue>::TVector& newg, const typename Types<TValue>::TVector& oldg)
    {
        auto s = newx - oldx;
        auto y = newg - oldg;
        return s.dot(s) / s.dot(y);
    }

    template<typename TValue>
    TValue SPGMinimize(ProjectableProblem<TValue> &problem,
                       typename ProjectableProblem<TValue>::TVector& x,
                       size_t maxNumIt  = 5000,
                       TValue geps      = (TValue) 1e-3,
                       bool verbose     = true,
                       bool fixedNumIt  = false,
                       size_t maxSrchIt = 100,
                       TValue gamma     = (TValue) 1e-4,
                       TValue maxAlpha  = (TValue) 1e4,
                       TValue minAlpha  = (TValue) 1e-10)
    {
        typedef typename Types<TValue>::TVector TVector;
        const size_t dim = problem.Dimensions();
        TVector g(dim), candx(dim), candg(dim), d(dim);
        problem.ProvideStartingPoint(x);
        TValue f      = problem.Eval(x, g);
        TValue gnorm  = ProjectedGradientNorm(problem, x, g);
        TValue alpha  = TValue(1.0) / (gnorm * x.norm());
        TValue gTd    = SPGComputeDirection(problem, x, g, alpha, d);
        size_t t      = 1;
        size_t fevals = 1;
        problem.Report("SPG: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);

        while(gnorm > geps && t < maxNumIt)
        {
            TValue lambda = (TValue) 1.0;
            bool accepted = false;
            size_t srchIt = 0;

            do
            {
                TVector candx   = x + lambda * d;
                TValue  candf   = problem.Eval(candx, candg);
                TValue  suffdec = gamma * lambda * gTd;

                if(srchIt > 0 && verbose)
                    problem.Report("SPG:    SrchIt %4d: f %-10.8f t %-10.8f\n", srchIt, candf, alpha * lambda);

                if(candf <= f + suffdec)
                {
                    alpha    = std::min(maxAlpha, std::max(minAlpha, SPGComputeStepSize(problem, candx, x, candg, g)));
                    f        = candf;
                    accepted = true;
                    x        = candx;
                    g        = candg;
                }
                else if(srchIt >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda  *= 0.5;
                    srchIt++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchIt >= maxSrchIt)
            {
                problem.Report("SPG: Linesearch cannot make further progress.\n");
                break;
            }

            if((! fixedNumIt) || (t % 10 == 0 || t == maxNumIt))
                gnorm = ProjectedGradientNorm(problem, x, g);

            gTd   = SPGComputeDirection(problem, x, g, alpha, d);

            if(verbose && !fixedNumIt)
                problem.Report("SPG: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);

            t++;
        }

        problem.Report("SPG: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    template<typename TConstrainedQuadratic>
    typename TConstrainedQuadratic::TValue
    SPGMinimizeCQ(TConstrainedQuadratic &problem,
                  typename TConstrainedQuadratic::TVector& x,
                  size_t maxNumIt  = 5000,
                  typename TConstrainedQuadratic::TValue geps = 1e-3,
                  bool verbose     = true,
                  bool fixedNumIt  = false,
                  size_t maxSrchIt = 100,
                  typename TConstrainedQuadratic::TValue gamma     = 1e-4,
                  typename TConstrainedQuadratic::TValue maxAlpha  = 1e4,
                  typename TConstrainedQuadratic::TValue minAlpha  = 1e-10)
    {
        typedef typename TConstrainedQuadratic::TVector TVector;
        typedef typename TConstrainedQuadratic::TValue  TValue;
        std::deque<TValue> f_hist;
        const size_t dim = problem.Dimensions();
        TVector g(dim), candx(dim), candg(dim), d(dim), Qd(dim);
        problem.ProvideStartingPoint(x);
        TValue f      = problem.Eval(x, g);
        TValue gnorm  = ProjectedGradientNorm(problem, x, g);
        TValue alpha  = TValue(1.0) / (gnorm * x.norm());
        TValue gTd    = SPGComputeDirection(problem, x, g, alpha, d);
        size_t t      = 1;
        size_t fevals = 1;
        //problem.Report("SPGCQ: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);

        f_hist.push_back(f);
        while(gnorm > geps && t < maxNumIt)
        {
            TValue lambda = 1.0;
            bool accepted = false;
            size_t srchIt = 0;

            do
            {
                TVector candx   = x + lambda * d;
                TValue  candf   = problem.Eval(candx, candg);
                TValue  suffdec = gamma * lambda * gTd;

                //if(srchIt > 0 && verbose)
                //    problem.Report("SPGCQ:    SrchIt %4d: f %-10.8f t %-10.8f\n", srchIt, candf, alpha * lambda);

                if(candf <= *std::max_element(f_hist.begin(), f_hist.end()) + suffdec)
                {
                    alpha    = std::min(maxAlpha, std::max(minAlpha, SPGComputeStepSize(problem, candx, x, candg, g)));
                    f        = candf;
                    accepted = true;
                    x        = candx;
                    g        = candg;
                    f_hist.push_back(f);
                    if( f_hist.size() > 8 )
                        f_hist.pop_front();
                }
                else if(srchIt >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda  *= 0.01;
                    srchIt++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchIt >= maxSrchIt)
            {
                //problem.Report("SPGCQ: Linesearch cannot make further progress.\n");
                break;
            }

            if((! fixedNumIt) || (t % 10 == 0 || t == maxNumIt))
                gnorm = ProjectedGradientNorm(problem, x, g);

            gTd   = SPGComputeDirection(problem, x, g, alpha, d);

            // if(verbose && !fixedNumIt)
            //    problem.Report("SPGCQ: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);

            t++;
        }

        //problem.Report("SPGCQ: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    // Compact representation of an n x n Hessian, maintained via L-BFGS updates
    template <typename TValue, size_t m>
    class CompactHessian
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, Eigen::Dynamic> TMatrix;

    private:
        TValue sigma;
        std::deque<TVector> Y;
        std::deque<TVector> S;
        Eigen::FullPivHouseholderQR< TMatrix > MQr;
#if 0
        TMatrix N;
#endif
        // Returns the product of the tranpose of N (a 2k x n matrix)
        // with an n x 1 vector v. We never instantiate N explicitly to save memory.
        const TVector NTv(const TVector& v) const
        {
            const int k = (int) Y.size();
            TVector ntv(2 * k);
            //#pragma omp parallel
            {
                //#pragma omp for nowait

                for(int i = 0; i < k; ++i)
                    ntv[i] = sigma * S[i].dot(v);

                //#pragma omp for

                for(int i = k; i < 2 * k; ++i)
                    ntv[i] = Y[i - k].dot(v);
            }
            return ntv;
        }

        // Returns the product of N (a n x 2k matrix) with a 2k x 1 vector v
        const TVector Nv(const TVector& v) const
        {
            const int n = (int) Y.front().size(), k = (int) Y.size();
            TVector nv1 = TVector::Zero(n), nv2 = TVector::Zero(n);
            //#pragma omp parallel sections
            {
                //#pragma omp section

                for(int i = 0; i < k; ++i)
                    nv1 += v[i] * sigma * S[i];

                //#pragma omp section

                for(int i = k; i < 2 * k; ++i)
                    nv2 += v[i] * Y[i - k];
            }
            return nv1 + nv2;
        }


#if 0
        // Returns the product of N (a n x 2k matrix) with a 2k x 1 vector v
        const TVector Nv(const TVector& v) const
        {
            const int n = (int) Y.front().size(), k = (int) Y.size();
            TVector nv = TVector::Zero(n), threadnv = TVector::Zero(n);
            #pragma omp parallel firstprivate(threadnv) shared(nv)
            {
                #pragma omp for

                for(int i = 0; i < k; ++i)
                    threadnv += (v[i] * sigma) * S[i] + v[k + i] * Y[i];

                #pragma omp critical
                {
                    nv += threadnv;
                }
            }
            return nv;
        }
#endif
    public:

        TVector Times(const TVector& v) const
        {
            if(Y.empty())
                return v;
            else
                return sigma * v - Nv(MQr.solve(NTv(v)));
        }

        void Update(const TVector& y, const TVector& s)
        {
            // Compute scaling factor for initial Hessian, which we choose as
            const TValue yTs = y.dot(s);

            if(yTs < 1e-12)   // Ensure B remains strictly positive definite
                return;

            if(Y.size() >= m)
            {
                Y.pop_front();
                S.pop_front();
            }

            Y.push_back(y);
            S.push_back(s);
            sigma = TValue(1.0) / (yTs / y.dot(y));
            const size_t k = Y.size(), n = Y.front().size();
            // D_k is the k x k diagonal matrix D_k = diag [s_0^Ty_0, ...,s_{k-1}^Ty_{k-1}].
            TVector minusd(k);
            //#pragma omp parallel for

            for(int i = 0; i < (int) k; ++ i)
                minusd[i] = - S[i].dot(Y[i]);

            const auto minusD = minusd.asDiagonal();
            // L_k is the k x k matrix with (L_k)_{i,j} = if( i > j ) s_i^T y_j else 0
            // (this is a lower triangular matrix with the main diagonal set to all zeroes)
            TMatrix L = TMatrix::Zero(k, k);
            //#pragma omp parallel for

            for(int j = 0; j < (int) k; ++j)
                for(size_t i = j + 1; i < k; ++i)
                    L(i, j) = S[i].dot(Y[j]);

            // S_k^T S_k is the symmetric k x k matrix with element (i,j) given by <s_i, s_j>
            TMatrix STS(k, k);
            //#pragma omp parallel for

            for(int j = 0; j < (int) k; ++j)
            {
                for(size_t i = j; i < k; ++i)
                {
                    const TValue sTs = S[i].dot(S[j]);
                    STS(i, j) = sTs;
                    STS(j, i) = sTs;
                }
            }

            // M is the 2k x 2k matrix given by: M = [ \sigma * S_k^T S_k    L_k ]
            //                                       [         L_k^T        -D_k ]
            TMatrix M(2 * k, 2 * k);
            M.topLeftCorner(k, k)     = sigma * STS;
            M.bottomLeftCorner(k, k)  = L.transpose();
            M.topRightCorner(k, k)    = L;
            M.bottomRightCorner(k, k) = minusD;
            // Save QR decomposition of M for later use in left-multiplication by M^{-1}
            MQr = M.fullPivHouseholderQr();
#if 0
            // N is the n x 2k matrix given by: N = [ \sigma * s_1  ... \sigma * s_k  y_1 ... y_k ],
            // where s_i and y_i are n x 1 column vectors.
            N.resize(n, 2 * k);

            for(int j = 0; j < k; ++j)
            {
                N.col(j)   = sigma * S[j];
                N.col(k + j) = Y[j];
            }

#endif
        }
    };

    template <typename TValue, size_t M>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> operator * (const CompactHessian<TValue, M>& B,
            const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& v)
    {
        return B.Times(v);
    }

    // Forms a quadratic model around fun, the argmin of which then determines a feasible
    // quasi-Newton descent direction
    template <typename TValue, size_t M>
    class PQNSubproblem : public ProjectableProblem<TValue>
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;

    private:
        ProjectableProblem<TValue>& fun;
        const TValue f_k;
        const TVector& x_k;
        const TVector& g_k;
        const CompactHessian<TValue, M>& B_k;

    public:
        PQNSubproblem(ProjectableProblem<TValue>& fun_,
                      TValue f_k_,
                      const TVector& x_k_,
                      const TVector& g_k_,
                      const CompactHessian<TValue, M>& B_k_) : fun(fun_), f_k(f_k_), x_k(x_k_), g_k(g_k_), B_k(B_k_)
        {
        }

        unsigned Dimensions() const
        {
            return fun.Dimensions();
        }

        // Compute objective and gradient of the quadratic model at the current iterate:
        //  q_k(p)         = f_k + (p-x_k)^T g_k + 1/2 (p-x_k)^T B_k(p-x_k)
        //  \nabla q_k(p)  = g_k + B_k(p-x_k)
        TValue Eval(const TVector& p, TVector& nabla)
        {
            const TVector d  = p - x_k;
            const TVector Bd = B_k * d;
            const TValue q_k = f_k + d.dot(g_k) + 0.5 * d.dot(Bd);
            nabla = g_k + Bd;
            return q_k;
        }

        TVector Project(const TVector& point) const
        {
            return fun.Project(point);
        }

        bool IsFeasible(const TVector& point) const
        {
            return fun.IsFeasible(point);
        }

        void ProvideStartingPoint(TVector& point) const
        {
            point = Project(x_k);
        }

        virtual TValue Norm(const TVector& g) const
        {
            return fun.Norm(g);
        }
    };

    template <size_t M, typename TValue>
    TValue PQNMinimize(ProjectableProblem<TValue>& prob, Eigen::Matrix<TValue, Eigen::Dynamic, 1>& x,
                       size_t maxNumIt   = 1000,
                       TValue optTol     = TValue(1e-3),
                       size_t numInnerIt = 50,
                       bool verbose      = true,
                       size_t maxSrchIt  = 100,
                       TValue gamma      = TValue(1e-6))
    {
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        const unsigned Dim = prob.Dimensions();
        TVector candx(Dim), g(Dim), candg(Dim), d(Dim), s(Dim), y(Dim);
        prob.ProvideStartingPoint(x);
        TValue f     = prob.Eval(x, g), candf = TValue(0.0), suffdec = TValue(0.0);
        TValue gnorm = ProjectedGradientNorm(prob, x, g);
        size_t fevals = 1;
        size_t t      = 1;
        prob.Report("PQN: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);
        CompactHessian<TValue, M> B;

        while(gnorm > optTol && t < maxNumIt)
        {
            // Find descent direction
            if(t == 1)
            {
                // Initial direction, plain steepest descent
                d = prob.Project(x - 1e-4 * g / gnorm) - x;
            }
            else
            {
                // Update the limited-memory BFGS approximation to the Hessian
                B.Update(y, s);
                // Solve the quadratic subproblem approximately; we use the current iterate x as a guess
                // (note that this guarantees d being a descent direction if we perform at least
                //  one successful step of SPG - see Schmidt et al.)
                PQNSubproblem<TValue, M> subprob(prob, f, x, g, B);
                SPGMinimize(subprob, d, numInnerIt, optTol / TValue(10.0), false, true);
                d -= x;
            }

            // Backtracking line-search
            bool   accepted = false;
            TValue lambda   = TValue(1.0);
            TValue gTd      = g.dot(d);
            size_t srchit   = 0;

            do
            {
                candx   = x + lambda * d;
                candf   = prob.Eval(candx, candg);
                suffdec = gamma * lambda * gTd;

                if(srchit > 0 && verbose)
                    prob.Report("PQN:   SrchIt %4d: f %-10.8f t %-10.8f\n", srchit, candf, lambda);

                if(candf < f + suffdec)
                {
                    s = candx - x;
                    y = candg - g;
                    x = candx;
                    g = candg;
                    f = candf;
                    accepted = true;
                }
                else if(srchit >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda *= 0.5;
                    srchit++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchit >= maxSrchIt)
            {
                prob.Report("PQN: Line search cannot make further progress");
                break;
            }

            gnorm = ProjectedGradientNorm(prob, x, g);

            if(verbose)
                prob.Report("PQN: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);

            t++;
        }

        prob.Report("PQN: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    // Represents a linear system Ax = b.
    template<typename TValue>
    class LinearSystem
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> VectorType;

        // Store the right-hand side of the linear system, i.e. vector b, in the provided argument
        virtual void ProvideRightHandSide(VectorType& b) const = 0;

        // Store the inverse of the diagonal of the system matrix in the provided argument
        virtual void ProvideInverseDiagonal(VectorType& invDiag) const = 0;

        // Compute y = Ax and store the result in the provided output argument
        virtual void MultiplySystemMatrixBy(VectorType& y, const VectorType& x) const = 0;

        // Returns the number of components of b (or, equivalently, of x)
        virtual unsigned Dimensions() const = 0;

        virtual void Report(const char* fmt, ...) const
        {
            va_list args;
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
        }
    };

    template<typename TValue>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> CGSolve(const LinearSystem<TValue>& system,
            unsigned maxNumIt = 5000,
            TValue breakEps = 1e-6,
            bool verbose = true,
            bool *converged = NULL)
    {
        const unsigned Dim = system.Dimensions();
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        TVector Ap(Dim);
        TVector r(Dim);
        system.ProvideRightHandSide(r);
        TVector p     = r;
        TValue  rsold = r.dot(r);
        TVector x     = TVector::Zero(Dim);
        system.Report("CG: Initially  : ||r|| %-10.6f dim %u\n", sqrt(rsold), Dim);

        if( converged )
            *converged = true;

        if(sqrt(rsold) < breakEps)
            return x;

        unsigned t;

        for(t = 1; t <= maxNumIt; ++t)
        {
            system.MultiplySystemMatrixBy(Ap, p);
            TValue alpha = rsold / p.dot(Ap);
            x += alpha *  p;
            r -= alpha * Ap;
            const TValue rsnew = r.dot(r);

            if(verbose)
                system.Report("CG: MainIt %4d: ||r|| %-10.6f\n", t, sqrt(rsnew));

            if(sqrt(rsnew) < breakEps)
            {
                rsold = rsnew;
                break;
            }

            p *= rsnew / rsold;
            p += r;
            rsold = rsnew;
        }
        if( t > maxNumIt ) {
            if( converged )
                *converged = false;
        }

        system.Report("CG: FinIt  %4d: ||r|| %-10.6f\n", t, sqrt(rsold));
        return x;
    }

    template<typename TValue>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> PCGSolve(const LinearSystem<TValue>& system,
            unsigned maxNumIt = 5000,
            TValue breakEps = 1e-6,
            bool verbose = true,
            bool *converged = NULL)
    {
        const unsigned Dim = system.Dimensions();
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        TVector Minv_(Dim);
        system.ProvideInverseDiagonal(Minv_);
        const auto Minv = Minv_.asDiagonal();
        TVector Ap(Dim);
        TVector r(Dim);
        system.ProvideRightHandSide(r);
        TVector z = Minv * r;
        TVector p = z;
        TValue  rsold = r.dot(z);
        TValue  nrm   = r.norm();
        TVector x     = TVector::Zero(Dim);
        system.Report("PCG: Initially  : ||r|| %-10.6f dim %u\n", nrm, Dim);

        if( converged )
            *converged = true;

        if(sqrt(rsold) < breakEps)
            return x;

        unsigned t;

        for(t = 1; t <= maxNumIt; ++t)
        {
            system.MultiplySystemMatrixBy(Ap, p);
            TValue alpha = rsold / p.dot(Ap);
            x += alpha *  p;
            r -= alpha * Ap;

            nrm = r.norm();
            if(verbose)
                system.Report("PCG: MainIt %4d: ||r|| %-10.6f\n", t, nrm);
            if(nrm < breakEps)
                break;

            z = Minv * r;
            const TValue rsnew = r.dot(z);

            p *= rsnew / rsold;
            p += z;
            rsold = rsnew;
        }
        if( t > maxNumIt ) {
            if( converged )
                *converged = false;
        }

        system.Report("PCG: FinIt  %4d: ||r|| %-10.6f\n", t, nrm);
        return x;
    }

    template<typename TValue, size_t VarDim>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1>
    ScaleByBlockDiagonal(const Eigen::Matrix<TValue, Eigen::Dynamic, VarDim>& D,
                         const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& v)
    {
        Eigen::Matrix<TValue, Eigen::Dynamic, 1> ret(v.size());
        const auto NumBlocks = v.size() / VarDim;

        #pragma omp parallel for
        for( int i = 0; i < NumBlocks; ++i )
        {
            ret.template segment<VarDim>(VarDim*i) = D.template block<VarDim, VarDim>(VarDim*i, 0) * v.template segment<VarDim>(VarDim*i);
        }
        return ret;
    }

    template<typename TLinearSystem>
    typename TLinearSystem::VectorType BlockPCGSolve(const TLinearSystem &system,
            unsigned maxNumIt = 5000,
            typename TLinearSystem::TValue breakEps = 1e-6,
            bool verbose = true,
            bool *converged = NULL)
    {
        typedef typename TLinearSystem::TValue TValue;
        typedef typename TLinearSystem::VectorType TVector;
        typedef typename TLinearSystem::BlockDiagonalType TBlockDiagonal;
        static const size_t VarDim = TLinearSystem::Dim;

        const auto Dim = system.Dimensions();

        TBlockDiagonal Minv(Dim, VarDim);
        system.ProvideInverseBlockDiagonal(Minv);
        TVector Ap(Dim);
        TVector r(Dim);
        system.ProvideRightHandSide(r);
        TVector z = ScaleByBlockDiagonal<TValue, VarDim>(Minv, r);
        TVector p = z;
        TValue  rsold = r.dot(z);
        TValue  nrm   = r.norm();
        TVector x     = TVector::Zero(Dim);
        system.Report("PCG: Initially  : ||r|| %-10.6f dim %u\n", nrm, Dim);

        if( converged )
            *converged = true;

        if(sqrt(rsold) < breakEps)
            return x;

        unsigned t;

        for(t = 1; t <= maxNumIt; ++t)
        {
            system.MultiplySystemMatrixBy(Ap, p);
            TValue alpha = rsold / p.dot(Ap);
            x += alpha *  p;
            r -= alpha * Ap;

            nrm = r.norm();
            if(verbose)
                system.Report("PCG: MainIt %4d: ||r|| %-10.6f\n", t, nrm);
            if(nrm < breakEps)
                break;

            z = ScaleByBlockDiagonal<TValue, VarDim>(Minv, r);
            const TValue rsnew = r.dot(z);

            p *= rsnew / rsold;
            p += z;
            rsold = rsnew;
        }
        if( t > maxNumIt ) {
            if( converged )
                *converged = false;
        }

        system.Report("PCG: FinIt  %4d: ||r|| %-10.6f\n", t, nrm);
        return x;
    }

    template <typename TValue, size_t m>
    class CompactInverseHessian
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;

    private:
        std::deque<TVector> Y;
        std::deque<TVector> S;

    public:
        bool Empty() const
        {
            return Y.empty();
        }

        void Clear()
        {
            Y.clear();
            S.clear();
        }

        void Update(const TVector& y, const TVector& s)
        {
            // Compute scaling factor for initial Hessian, which we choose as
            const TValue yTs = y.dot(s);

            if(yTs < 1e-12)   // Ensure B remains strictly positive definite
                return;

            if(Y.size() >= m)
            {
                Y.pop_front();
                S.pop_front();
            }

            Y.push_back(y);
            S.push_back(s);
        }

        // Returns the product of vector g with the compact inverse Hessian;
        // this is computed using the classic two-loop LBFGS formula.
        TVector Times(const TVector& v) const
        {
            const size_t k = Y.size();

            if(k == 0)
                return v;

            TVector p(v), alphas(k);

            for(int i = (int) k - 1; i  >= 0; --i)
            {
                const auto alpha = S[i].dot(p) / S[i].dot(Y[i]);
                p -= alpha * Y[i];
                alphas[i] = alpha;
            }

            p *= S.back().dot(Y.back()) / Y.back().squaredNorm();

            for(size_t i = 0; i < k; ++i)
            {
                const auto beta = Y[i].dot(p) / Y[i].dot(S[i]);
                p += (alphas[i] - beta) * S[i];
            }

            return p;
        }
    };

    template <typename TValue, size_t M>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> operator * (const CompactInverseHessian<TValue, M>& H,
            const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& v)
    {
        return H.Times(v);
    }

    template <size_t M, typename TValue>
    TValue LBFGSMinimize(UnconstrainedProblem<TValue>& prob, Eigen::Matrix<TValue, Eigen::Dynamic, 1>& x,
                         size_t maxNumIt   = 1000,
                         TValue optTol     = TValue(1e-3),
                         bool verbose      = true,
                         size_t maxSrchIt  = 10,
                         TValue gamma      = TValue(1e-6))
    {
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        const unsigned Dim = prob.Dimensions();
        TVector candx(Dim), g(Dim), candg(Dim), d(Dim), s(Dim), y(Dim);
        prob.ProvideStartingPoint(x);
        TValue f     = prob.Eval(x, g), candf = TValue(0.0), suffdec = TValue(0.0);
        TValue gnorm = prob.Norm(g);
        size_t fevals = 1;
        size_t t      = 1;
        prob.Report("LBFGS: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);
        CompactInverseHessian<TValue, M> H;

        while(gnorm > optTol && t < maxNumIt)
        {
            // Find descent direction
            if(H.Empty())
                d = - g / gnorm; // Initial direction, plain steepest descent
            else
                d = - (H * g);   // Scale by inverse Hessian

            // Backtracking line-search
            bool   accepted = false;
            TValue lambda   = TValue(1.0);
            TValue gTd      = g.dot(d);
            size_t srchit   = 0;

            do
            {
                candx   = x + lambda * d;
                candf   = prob.Eval(candx, candg);
                suffdec = gamma * lambda * gTd;

                if(srchit > 0 && verbose)
                    prob.Report("LBFGS:   SrchIt %4d: f %-10.8f t %-10.8f\n", srchit, candf, lambda);

                if(candf < f + suffdec)
                {
                    s = candx - x;
                    y = candg - g;
                    x = candx;
                    g = candg;
                    f = candf;
                    accepted = true;
                }
                else if(srchit >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda *= 0.5;
                    srchit++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchit >= maxSrchIt)
            {
                prob.Report("LBFGS: Line search cannot make further progress\n");
                break;
            }

            // Valid step - update the L-BFGS approximation to the inverse Hessian
            H.Update(y, s);
            gnorm = prob.Norm(g);

            if(verbose)
                prob.Report("LBFGS: MainIt %4d: f %-10.8f ||g|| %-10.4f\n", t, f, gnorm);

            t++;
        }

        prob.Report("LBFGS: FinIt  %4d: f %-10.4f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    template <size_t M, typename TValue>
    TValue RestartingLBFGSMinimize(ProjectableProblem<TValue>& prob, Eigen::Matrix<TValue, Eigen::Dynamic, 1>& x,
                                   size_t maxNumIt   = 1000,
                                   TValue optTol     = TValue(1e-3),
                                   bool verbose      = true,
                                   size_t maxSrchIt  = 10,
                                   TValue gamma      = TValue(1e-6))
    {
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        const unsigned Dim = prob.Dimensions();
        TVector candx(Dim), g(Dim), candg(Dim), d(Dim), s(Dim), y(Dim);
        prob.ProvideStartingPoint(x);
        TValue f     = prob.Eval(x, g), candf = TValue(0.0), suffdec = TValue(0.0);
        TValue gnorm = ProjectedGradientNorm(prob, x, g);
        size_t fevals = 1;
        size_t t      = 1;
        prob.Report("LBFGS: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);
        //Sleep(3000);
        CompactInverseHessian<TValue, M> H;

        while(gnorm > optTol && t < maxNumIt)
        {
            // Find descent direction
            if( H.Empty() )//if(t == 1)
            {
                // Initial direction, plain steepest descent
                d = prob.Project(x - g * (1e-3 / gnorm)) - x;
            }
            else
            {
                // Scale the gradient by the L-BFGS approximation to the inverse Hessian
                d = prob.Project(x - (H * g)) - x;
            }

            // Backtracking line-search
            bool   accepted = false;
            TValue lambda   = TValue(1.0);
            TValue gTd      = g.dot(d);
            size_t srchit   = 0;

            do
            {
                candx   = x + lambda * d;
                candf   = prob.Eval(candx, candg);
                suffdec = gamma * lambda * gTd;

                if(srchit > 0 && verbose) {
                    prob.Report("LBFGS:   SrchIt %4d: f %-10.8f t %-10.8f\n", srchit, candf, lambda);
                    //Sleep(1000);
                }

                if(candf < f + suffdec)
                {
                    s = candx - x;
                    y = candg - g;
                    x = candx;
                    g = candg;
                    f = candf;
                    accepted = true;
                    srchit = 0;
                }                                               // Allow more line search steps initially
                else if((!H.Empty() && srchit >= maxSrchIt) || (H.Empty() && srchit >= maxSrchIt * 10))
                {
                    accepted = true;
                    H.Clear();
                    srchit++;
                }
                else
                {
                    lambda *= 0.125;
                    srchit++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchit >= maxSrchIt * 10)
            {
                prob.Report("LBFGS: Line search cannot make further progress\n");
                break;
            }

            if(srchit > maxSrchIt)
            {
                prob.Report("LBFGS: Projected direction is bad - resetting approximation to inverse Hessian\n");
                continue;
            }

            // Valid step - update the L-BFGS approximation to the inverse Hessian
            H.Update(y, s);
            gnorm = ProjectedGradientNorm(prob, x, g);

            if(verbose) {
                prob.Report("LBFGS: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);
                //Sleep(1000);
            }

            t++;
        }

        prob.Report("LBFGS: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        //Sleep(5000);
        return f;
    }

    // Ternary search algorithm: Simple method for 1d-optimization
    //
    // Can be used to solve one-dimensional linesearch problems efficiently and without
    // the derivative with respect to the parameter. Linear convergence.
    template<typename TValue, typename TFunction>
    TValue TernarySearch(TFunction f, TValue left, TValue right, TValue absolutePrecision, bool verbose = false)
    {
        // left and right are the current bounds; the minimum is between them
        if((right - left) < absolutePrecision)
            return (left + right) / TValue(2);

        const auto leftThird = (TValue(2) * left + right) / TValue(3);
        const auto rightThird = (left + TValue(2) * right) / TValue(3);
        auto flt = f(leftThird);
        auto frt = f(rightThird);

        if(verbose)
        {
            std::cout << "  TernarySearch: f(" << leftThird << ")=" << flt
                      << "  f(" << rightThird << ")=" << frt << std::endl;
        }

        if(flt > frt)
            return TernarySearch(f, leftThird, right, absolutePrecision);
        else
            return TernarySearch(f, left, rightThird, absolutePrecision);
    }

    template<typename TValue, typename TFunction>
    TValue SafeDescentSearch(TFunction f, TValue left, TValue right, bool verbose = false)
    {
        // 1. Assert the minimal stepsize is descent
        auto alpha = left;
        auto f_0 = f(0);
        auto f_alpha = f(alpha);

        if(f_alpha > f_0)
        {
            if(verbose)
            {
                std::cout << "   SafeDescentSearch: minimum step size " << left << " failed to provide descent."
                          << std::endl;
            }

            return std::numeric_limits<TValue>::signaling_NaN();
        }

        double growth_factor = std::sqrt(2.0);
        auto f_prev = f_alpha;
        auto alpha_prev = alpha;

        do
        {
            if(verbose)
            {
                std::cout << "   SafeDescentSearch: alpha=" << alpha << ", f(alpha)=" << f_alpha
                          << std::endl;
            }

            // Safe previous
            alpha_prev = alpha;
            f_prev = f_alpha;
            // Grow
            alpha *= growth_factor;

            if(alpha > right)
                break;

            f_alpha = f(alpha);
        }
        while(f_alpha < f_prev);

        return (alpha_prev);
    }

    /**
    * An implementation of the active Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    * for non-linear, non-convex, non-smooth, global function minimization.
    * The CMA-Evolution Strategy (CMA-ES) is a reliable stochastic optimization method
    * which should be applied if derivative-based methods, e.g. quasi-Newton BFGS or
    * conjugate gradient, fail due to a rugged search landscape (e.g. noise, local
    * optima, outlier, etc.) of the objective function. Like a
    * quasi-Newton method, the CMA-ES learns and applies a variable metric
    * on the underlying search space. Unlike a quasi-Newton method, the
    * CMA-ES neither estimates nor uses gradients, making it considerably more
    * reliable in terms of finding a good, or even close to optimal, solution.
    *
    * In general, on smooth objective functions the CMA-ES is roughly ten times
    * slower than BFGS (counting objective function evaluations, no gradients provided).
    * For up to N=10 variables also the derivative-free simplex
    * direct search method (Nelder and Mead) can be faster, but it is
    * far less reliable than CMA-ES.
    *
    * The CMA-ES is particularly well suited for non-separable
    * and/or badly conditioned problems. To observe the advantage of CMA compared
    * to a conventional evolution strategy, it will usually take about
    * 30 N function evaluations. On difficult problems the complete
    * optimization (a single run) is expected to take roughly between
    * 30 N and 300 N^2 function evaluations.</p>
    *
    * This implementation is translated and adapted from the Java version
    * contained in Apache Commons Math, which is in turn based on the
    * Matlab reference implementation cmaes.m, version 3.51.
    *
    * For more information, please refer to the following links:
    *
    *  http://svn.apache.org/viewvc/commons/proper/math/trunk/src/main/java/org/apache/commons/math/optimization/direct/CMAESOptimizer.java?revision=1212377&view=markup
    *  http://www.lri.fr/~hansen/cmaes.m
    *  http://www.lri.fr/~hansen/cmaesintro.html
    *  http://en.wikipedia.org/wiki/CMA-ES/
    */
    template<typename TValue>
    class CMAESOptimizer
    {
    public:
        typedef TValue                                                 TScalar;
        typedef Eigen::Matrix<TScalar, Eigen::Dynamic, 1>              TVector;
        typedef Eigen::Matrix<int, Eigen::Dynamic, 1>                  TIntVector;
        typedef Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> TMatrix;
        typedef std::pair<TVector, TValue>                             TSolution;

        /** Default value for checkFeasableCount */
        static const int DEFAULT_CHECKFEASABLECOUNT = 0;

        /** Default value for stopFitness */
        static TScalar DEFAULT_STOPFITNESS()
        {
            return 0;
        }

        /** Default value for isActiveCMA */
        static const bool DEFAULT_ISACTIVECMA = true;

        /** Default value for maxIterations */
        static const int DEFAULT_MAXITERATIONS = 30000;

        /** Default value for maxEvaluations */
        static const int DEFAULT_MAXEVALUATIONS = 300000;

        /** Default value for diagonalOnly */
        static const int DEFAULT_DIAGONALONLY = 0;

        /** Whether statistics are collected */
        static const bool DEFAULT_GENERATESTATISTICS = false;

        /** Default user-defined convergence checker */
        static bool DEFAULT_CHECKER(int, const TSolution&, const TSolution&)
        {
            return false;
        }

        /** Declare inner classes */
        class FitnessFunction;
        friend class CMAESOptimizer<TValue>::FitnessFunction;

        // global search parameters
        /**
        * Population size, offspring number. The primary strategy parameter to play
        * with, which can be increased from its default value. Increasing the
        * population size improves global search properties in exchange to speed.
        * Speed decreases, as a rule, at most linearly with increasing population
        * size. It is advisable to begin with the default small population size.
        */
        int lambda; // population size
        /**
        * Covariance update mechanism, default is active CMA. isActiveCMA = true
        * turns on "active CMA" with a negative update of the covariance matrix and
        * checks for positive definiteness. OPTS.CMA.active = 2 does not check for
        * pos. def. and is numerically faster. Active CMA usually speeds up the
        * adaptation.
        */
        bool isActiveCMA;
        /**
        * Determines how often a new random offspring is generated in case it is
        * not feasible / beyond the defined limits, default is 0. Only relevant if
        * boundaries != NULL.
        */
        int checkFeasableCount;
        /**
        * Lower and upper boundaries of the objective variables. boundaries.size() == 0
        * means no boundaries.
        */
        TVector lowerBoundaries;
        TVector upperBoundaries;
        /**
        * Individual sigma values - initial search volume. inputSigma determines
        * the initial coordinate wise standard deviations for the search. Setting
        * SIGMA one third of the initial search region is appropriate.
        */
        TVector inputSigma;
        /** Number of objective variables/problem dimension */
        int dimension;
        /**
        * Defines the number of initial iterations, where the covariance matrix
        * remains diagonal and the algorithm has internally linear time complexity.
        * diagonalOnly = 1 means keeping the covariance matrix always diagonal and
        * this setting also exhibits linear space complexity. This can be
        * particularly useful for dimension > 100.
        * See http://hal.archives-ouvertes.fr/inria-00287367/en - A Simple Modification in CMA-ES
        */
        int diagonalOnly;
        /** Indicates whether statistic data is collected. */
        bool generateStatistics;

        // termination criteria
        /** Maximal number of iterations allowed. */
        int maxIterations;
        /** Maximal number of function evaluations allowed. */
        int maxEvaluations;
        /** Limit for fitness value. */
        TScalar stopFitness;
        /** Stop if x-changes larger stopTolUpX. */
        TScalar stopTolUpX;
        /** Stop if x-change smaller stopTolX. */
        TScalar stopTolX;
        /** Stop if fun-changes smaller stopTolFun. */
        TScalar stopTolFun;
        /** Stop if back fun-changes smaller stopTolHistFun. */
        TScalar stopTolHistFun;

        // selection strategy parameters
        /** Number of parents/points for recombination. */
        int mu; //
        /** log(mu + 0.5), stored for efficiency. */
        TScalar logMu2;
        /** Array for weighted recombination. */
        TMatrix weights;
        /** Variance-effectiveness of sum w_i x_i. */
        TScalar mueff; //

        // dynamic strategy parameters and constants
        /** Overall standard deviation - search volume. */
        TScalar sigma;
        /** Cumulation constant. */
        TScalar cc;
        /** Cumulation constant for step-size. */
        TScalar cs;
        /** Damping for step-size. */
        TScalar damps;
        /** Learning rate for rank-one update. */
        TScalar ccov1;
        /** Learning rate for rank-mu update' */
        TScalar ccovmu;
        /** Expectation of ||N(0,I)|| == norm(randn(N,1)). */
        TScalar chiN;
        /** Learning rate for rank-one update - diagonalOnly */
        TScalar ccov1Sep;
        /** Learning rate for rank-mu update - diagonalOnly */
        TScalar ccovmuSep;

        // CMA internal values - updated each generation
        /** Objective variables. */
        TMatrix xmean;
        /** Evolution path. */
        TMatrix pc;
        /** Evolution path for sigma. */
        TMatrix ps;
        /** Norm of ps, stored for efficiency. */
        TScalar normps;
        /** Coordinate system. */
        TMatrix B;
        /** B*D, stored for efficiency. */
        TMatrix BD;
        /** Diagonal of sqrt(D), stored for efficiency. */
        TMatrix diagD;
        /** Covariance matrix. */
        TMatrix C;
        /** Diagonal of C, used for diagonalOnly. */
        TMatrix diagC;
        /** Number of iterations already performed. */
        int iterations;
        /** Number of function evaluations. */
        int evaluations;

        /** History queue of best values. */
        std::deque<TScalar> fitnessHistory;
        /** Size of history queue of best values. */
        int historySize;

        /** Random generator. */
        std::mt19937 random;
        std::normal_distribution<TValue> normal;

        /** Convergence checker. */
        std::function<bool (int, const TSolution&, const TSolution&)> checker;

        /** History of sigma values. */
        std::vector<TScalar> statisticsSigmaHistory;
        /** History of mean matrix. */
        std::vector<TMatrix> statisticsMeanHistory;
        /** History of fitness values. */
        std::vector<TScalar> statisticsFitnessHistory;
        /** History of D matrix. */
        std::vector<TMatrix> statisticsDHistory;

        /** Objective function. */
        std::function<TValue(const TVector&)> objective;

    public:
        /**
        * lambda              Population size.
        * inputSigma          Initial search volume; sigma of offspring objective variables.
        * boundaries          Boundaries for objective variables.
        * maxIterations       Maximal number of iterations.
        * maxEvaluations      Maximal number of function evaluations.
        * stopFitness         Whether to stop if objective function value is smaller than stopFitness.
        * isActiveCMA         Chooses the covariance matrix update method.
        * diagonalOnly        Number of initial iterations, where the covariance matrix remains diagonal.
        * checkFeasableCount  Determines how often new random objective variables are generated in case they are out of bounds.
        * random              Random generator.
        * generateStatistics  Whether statistic data is collected.
        * checker             Convergence checker.
        */
        CMAESOptimizer(int lambda_ = -1, const TVector& lowerBoundaries_ = TVector(), const TVector& upperBoundaries_ = TVector(),
                       int maxEvaluations_ = DEFAULT_MAXEVALUATIONS, const std::function<bool (int, const TSolution&, const TSolution&)>& checker_ = DEFAULT_CHECKER,
                       int maxIterations_ = DEFAULT_MAXITERATIONS, const TVector& inputSigma_ = TVector(), TValue stopFitness_ = DEFAULT_STOPFITNESS(),
                       bool isActiveCMA_ = DEFAULT_ISACTIVECMA, int diagonalOnly_ = DEFAULT_DIAGONALONLY, int checkFeasableCount_ = DEFAULT_CHECKFEASABLECOUNT,
                       const std::mt19937& random_ = std::mt19937(), bool generateStatistics_ = DEFAULT_GENERATESTATISTICS)
            :	checker(checker_), lambda(lambda_), inputSigma(inputSigma_), lowerBoundaries(lowerBoundaries_), upperBoundaries(upperBoundaries_),
                maxIterations(maxIterations_), maxEvaluations(maxEvaluations_), stopFitness(stopFitness_), isActiveCMA(isActiveCMA_), diagonalOnly(diagonalOnly_),
                checkFeasableCount(checkFeasableCount_), random(random_), generateStatistics(generateStatistics_)
        {
        }

        /**
        * objective   Objective function to be minimized.
        * startPoint  Initial guess of the solution.
        * returns the best solution (point, function value) found.
        */
        TSolution doOptimize(const std::function<TValue(const TVector&)>& objective_, const TVector& startPoint)
        {
            // -------------------- Initialization --------------------------------
            objective = objective_;
            FitnessFunction fitfun(this);
            TVector guess = fitfun.encode(startPoint);
            // number of objective variables/problem dimension
            dimension = guess.size();
            initializeCMA(guess);
            iterations = 0;
            evaluations = 0;
            TValue bestValue = fitfun.value(guess);
            push(fitnessHistory, bestValue);
            TSolution optimum = std::make_pair(startPoint, bestValue);
            TSolution lastResult = std::make_pair(TVector(), std::numeric_limits<TValue>::max());

            // -------------------- Generation Loop --------------------------------

            for(iterations = 1; iterations <= maxIterations; iterations++)
            {
                // Generate and evaluate lambda offspring
                TMatrix arz = randn1(dimension, lambda);
                TMatrix arx = zeros(dimension, lambda);
                TVector fitness(lambda);

                // generate random offspring
                for(int k = 0; k < lambda; k++)
                {
                    TMatrix arxk;

                    for(int i = 0; i < checkFeasableCount + 1; i++)
                    {
                        if(diagonalOnly <= 0)
                        {
                            arxk = xmean + (BD * arz.col(k) * sigma); // m + sig * Normal(0,C)
                        }
                        else
                        {
                            arxk = xmean + (times(diagD, arz.col(k)) * sigma);
                        }

                        if(i >= checkFeasableCount || fitfun.isFeasible(arxk.col(0)))
                        {
                            break;
                        }

                        // regenerate random arguments for row
                        arz.col(k) = randn(dimension);
                    }

                    copyColumn(arxk, 0, arx, k);

                    try
                    {
                        fitness[k] = fitfun.value(arx.col(k)); // compute fitness
                    }
                    catch(const std::exception&)      // too many function evaluations
                    {
                        goto breakGenerationLoop;
                    }
                }

                // Sort by fitness and compute weighted mean into xmean
                TIntVector arindex = sortedIndices(fitness);
                // Calculate new xmean, this is selection and recombination
                TMatrix xold = xmean; // for speed up of Eq. (2) and (3)
                TMatrix bestArx = selectColumns(arx, arindex.head(mu));
                xmean = bestArx * weights;
                TMatrix bestArz = selectColumns(arz, arindex.head(mu));
                TMatrix zmean = bestArz * weights;
                bool hsig = updateEvolutionPaths(zmean, xold);

                if(diagonalOnly <= 0)
                {
                    updateCovariance(hsig, bestArx, arz, arindex, xold);
                }
                else
                {
                    updateCovarianceDiagonalOnly(hsig, bestArz, xold);
                }

                // Adapt step size sigma - Eq. (5)
                sigma *= std::exp(std::min(1.0, (normps / chiN - 1.) * cs / damps));
                TValue bestFitness = fitness[arindex[0]];
                TValue worstFitness = fitness[arindex[arindex.size() - 1]];

                if(bestValue > bestFitness)
                {
                    bestValue = bestFitness;
                    lastResult = optimum;
                    optimum = std::make_pair(fitfun.decode(bestArx.col(0)), bestFitness);

                    if(lastResult.first.size() != 0)
                    {
                        if(checker(iterations, optimum, lastResult))
                        {
                            goto breakGenerationLoop;
                        }
                    }
                }

                // handle termination criteria
                // Break, if fitness is good enough
                if(stopFitness != 0)    // only if stopFitness is defined
                {
                    if(bestFitness < stopFitness)
                    {
                        goto breakGenerationLoop;
                    }
                }

                TVector sqrtDiagC = sqrt(diagC).col(0);
                TVector pcCol = pc.col(0);

                for(int i = 0; i < dimension; i++)
                {
                    if(sigma * (std::max(std::abs(pcCol[i]), sqrtDiagC[i])) > stopTolX)
                    {
                        break;
                    }

                    if(i >= dimension - 1)
                    {
                        goto breakGenerationLoop;
                    }
                }

                for(int i = 0; i < dimension; i++)
                {
                    if(sigma * sqrtDiagC[i] > stopTolUpX)
                    {
                        goto breakGenerationLoop;
                    }
                }

                TValue historyBest = *std::min_element(fitnessHistory.begin(), fitnessHistory.end());
                TValue historyWorst = *std::max_element(fitnessHistory.begin(), fitnessHistory.end());

                if(iterations > 2 && std::max(historyWorst, worstFitness) -
                        std::min(historyBest, bestFitness) < stopTolFun)
                {
                    goto breakGenerationLoop;
                }

                if(iterations > fitnessHistory.size() &&
                        historyWorst - historyBest < stopTolHistFun)
                {
                    goto breakGenerationLoop;
                }

                // condition number of the covariance matrix exceeds 1e14
                if(max(diagD) / min(diagD) > 1e7)
                {
                    goto breakGenerationLoop;
                }

                // user defined termination
                TSolution current = std::make_pair(bestArx.col(0), bestFitness);

                if(lastResult.first.size() != 0 &&
                        checker(iterations, current, lastResult))
                {
                    goto breakGenerationLoop;
                }

                lastResult = current;

                // Adjust step size in case of equal function values (flat fitness)
                if(bestValue == fitness[arindex[(int)(0.1 + lambda / 4.)]])
                {
                    sigma = sigma * std::exp(0.2 + cs / damps);
                }

                if(iterations > 2 && std::max(historyWorst, bestFitness) -
                        std::min(historyBest, bestFitness) == 0)
                {
                    sigma = sigma * std::exp(0.2 + cs / damps);
                }

                // store best in history
                push(fitnessHistory, bestFitness);
                fitfun.setValueRange(worstFitness - bestFitness);

                if(generateStatistics)
                {
                    statisticsSigmaHistory.push_back(sigma);
                    statisticsFitnessHistory.push_back(bestFitness);
                    statisticsMeanHistory.push_back(xmean.transpose());
                    statisticsDHistory.push_back(diagD.transpose() * 1E5);
                }
            }

breakGenerationLoop:
            return optimum;
        }

        /**
        * x  The point at which the function shall be evaluated.
        * returns the value of the objective function at the given point.
        * throws an exception if the maximum number of function evaluations has been exceeded.
        */
        TValue computeObjectiveValue(const TVector& x)
        {
            if(++evaluations > maxEvaluations)
                throw std::runtime_error("Maximum number of function evaluations exceeded.");

            return objective(x);
        }

        /**
        * Initialization of the dynamic search parameters
        *
        * guess  Initial guess for the arguments of the fitness function.
        */
        void initializeCMA(const TVector& guess)
        {
            if(lambda <= 0)
            {
                lambda = 4 + (int)(3.0 * std::log((TValue) dimension));
            }

            // initialize sigma
            TMatrix insigma(guess.size(), 1);

            for(int i = 0; i < guess.size(); i++)
            {
                const auto range = (lowerBoundaries.size() == 0) ? 1.0 : upperBoundaries[i] - lowerBoundaries[i];
                insigma(i, 0)    = ((inputSigma.size() == 0) ? 0.3 : inputSigma[i]) * range;
            }

            sigma = max(insigma); // overall standard deviation
            // initialize termination criteria
            stopTolUpX = TValue(1e3) * max(insigma);
            stopTolX = TValue(1e-11) * max(insigma);
            stopTolFun = TValue(1e-12);
            stopTolHistFun = TValue(1e-13);
            // initialize selection strategy parameters
            mu = lambda / 2; // number of parents/points for recombination
            logMu2 = std::log(mu + 0.5);
            weights = (log(sequence(1, mu, 1)) * -1.0).array() + logMu2;
            TValue sumw = 0;
            TValue sumwq = 0;

            for(int i = 0; i < mu; i++)
            {
                TValue w = weights(i, 0);
                sumw += w;
                sumwq += w * w;
            }

            weights *= (1.0 / sumw);
            mueff = sumw * sumw / sumwq; // variance-effectiveness of sum w_i x_i
            // initialize dynamic strategy parameters and constants
            cc = (4. + mueff / dimension) /
                 (dimension + 4. + 2. * mueff / dimension);
            cs = (mueff + 2.) / (dimension + mueff + 3.);
            damps = (1. + 2. * std::max(TValue(0), std::sqrt((mueff - TValue(1.)) /
                                        (dimension + TValue(1.))) - TValue(1.))) *
                    std::max(TValue(0.3), TValue(1.) - dimension /
                             (TValue(1e-6) + std::min(maxIterations, maxEvaluations / lambda))) + cs; // minor increment
            ccov1 = 2. / ((dimension + 1.3) * (dimension + 1.3) + mueff);
            ccovmu = std::min(TValue(1) - ccov1, TValue(2.) * (mueff - TValue(2.) + TValue(1.) / mueff) /
                              ((dimension + TValue(2.)) * (dimension + TValue(2.)) + mueff));
            ccov1Sep = std::min(TValue(1), ccov1 * (dimension + TValue(1.5)) / TValue(3.));
            ccovmuSep = std::min(TValue(1) - ccov1, ccovmu * (dimension + TValue(1.5)) / TValue(3.));
            chiN = std::sqrt((TValue) dimension) *
                   (1. - 1. / (4. * dimension) + 1 / (21. * dimension * dimension));
            // intialize CMA internal values - updated each generation
            xmean = guess; // objective variables
            diagD = insigma * (1. / sigma);
            diagC = square(diagD);
            pc = zeros(dimension, 1); // evolution paths for C and sigma
            ps = zeros(dimension, 1); // B defines the coordinate system
            normps = ps.norm();
            B = eye(dimension, dimension);
            BD = times(B, repmat(diagD.transpose(), dimension, 1));
            C = diag(diagC);
            historySize = 10 + (int)(3. * 10. * dimension / lambda);
            fitnessHistory.resize(historySize); // history of fitness values
            std::fill(fitnessHistory.begin(), fitnessHistory.end(), std::numeric_limits<TValue>::max());
        }

        /**
        * Update of the evolution paths ps and pc.
        *
        * zmean  Weighted row matrix of the gaussian random numbers generating
        *        the current offspring.
        * xold   xmean matrix of the previous generation.
        * returns hsig flag indicating a small correction.
        */
        bool updateEvolutionPaths(const TMatrix& zmean, const TMatrix& xold)
        {
            ps = (ps * (1. - cs)) +
                 ((B * (zmean)) * (std::sqrt(cs * (2. - cs) * mueff)));
            normps = ps.norm();
            bool hsig = (normps / std::sqrt(1. - std::pow(1. - cs, 2. * iterations)) / chiN)
                        < (1.4 + 2. / (dimension + 1.));
            pc *= (1. - cc);

            if(hsig)
            {
                pc += ((xmean - xold) * (std::sqrt(cc * (2. - cc) * mueff) / sigma));
            }

            return hsig;
        }

        /**
        * Update of the covariance matrix C for diagonalOnly > 0
        *
        * hsig     Flag indicating a small correction.
        * bestArz  Fitness-sorted matrix of the gaussian random values of the
        *          current offspring.
        * xold     xmean matrix of the previous generation.
        */
        void updateCovarianceDiagonalOnly(bool hsig,
                                          const TMatrix& bestArz,
                                          const TMatrix& xold)
        {
            // minor correction if hsig==false
            TValue oldFac = hsig ? 0 : ccov1Sep * cc * (2. - cc);
            oldFac += 1. - ccov1Sep - ccovmuSep;
            diagC = (diagC * oldFac) // regard old matrix
                    // plus rank one update
                    + (square(pc) * ccov1Sep)
                    // plus rank mu update
                    + ((times(diagC, square(bestArz) * weights)) * ccovmuSep);
            diagD = sqrt(diagC); // replaces eig(C)

            if(diagonalOnly > 1 && iterations > diagonalOnly)
            {
                // full covariance matrix from now on
                diagonalOnly = 0;
                B = eye(dimension, dimension);
                BD = diag(diagD);
                C = diag(diagC);
            }
        }

        /**
        * Update of the covariance matrix C.
        *
        * hsig     Flag indicating a small correction.
        * bestArx  Fitness-sorted matrix of the argument vectors producing the
        *          current offspring.
        * arz      Unsorted matrix containing the gaussian random values of the
        *          current offspring.
        * arindex  Indices indicating the fitness-order of the current offspring.
        * xold     xmean matrix of the previous generation.
        */
        void updateCovariance(bool hsig, const TMatrix& bestArx,
                              const TMatrix& arz, const TIntVector& arindex, const TMatrix& xold)
        {
            TValue negccov = 0;

            if(ccov1 + ccovmu > 0)
            {
                TMatrix arpos = (bestArx - repmat(xold, 1, mu)) * (1. / sigma); // mu difference vectors
                TMatrix roneu = (pc * pc.transpose()) * ccov1; // rank one update
                // minor correction if hsig==false
                TValue oldFac = hsig ? 0 : ccov1 * cc * (2. - cc);
                oldFac += 1. - ccov1 - ccovmu;

                if(isActiveCMA)
                {
                    // Adapt covariance matrix C active CMA
                    negccov = (1. - ccovmu) * 0.25 * mueff / (std::pow(dimension + 2., 1.5) + 2. * mueff);
                    TValue negminresidualvariance = TValue(0.66);
                    // keep at least 0.66 in all directions, small popsize are most critical
                    TValue negalphaold = 0.5; // where to make up for the variance
                    // loss,
                    // prepare vectors, compute negative updating matrix Cneg
                    TIntVector arReverseIndex = reverse(arindex);
                    TMatrix arzneg = selectColumns(arz, arReverseIndex.head(mu));
                    TMatrix arnorms = sqrt(sumRows(square(arzneg)));
                    TIntVector idxnorms = sortedIndices(arnorms.row(0));
                    TMatrix arnormsSorted = selectColumns(arnorms, idxnorms);
                    TIntVector idxReverse = reverse(idxnorms);
                    TMatrix arnormsReverse = selectColumns(arnorms, idxReverse);
                    arnorms = divide(arnormsReverse, arnormsSorted);
                    TIntVector idxInv = inverse(idxnorms);
                    TMatrix arnormsInv = selectColumns(arnorms, idxInv);
                    // check and set learning rate negccov
                    TValue negcovMax = (1. - negminresidualvariance) / TMatrix(square(arnormsInv) * weights)(0, 0);

                    if(negccov > negcovMax)
                    {
                        negccov = negcovMax;
                    }

                    arzneg = times(arzneg, repmat(arnormsInv, dimension, 1));
                    TMatrix artmp = BD * arzneg;
                    TMatrix Cneg =  artmp * diag(weights) * artmp.transpose();
                    oldFac += negalphaold * negccov;
                    C = (C * oldFac)
                        // regard old matrix
                        + roneu
                        // plus rank one update
                        + (arpos * (ccovmu + (1. - negalphaold) * negccov)
                           * (times(repmat(weights, 1, dimension), arpos.transpose())))
                        // plus rank mu update
                        - (Cneg * negccov);
                }
                else
                {
                    // Adapt covariance matrix C - nonactive
                    C = (C * oldFac) // regard old matrix
                        + roneu
                        // plus rank one update
                        + (arpos * ccovmu // plus rank mu update
                           * times(repmat(weights, 1, dimension), arpos.transpose()));
                }
            }

            updateBD(negccov);
        }

        /**
        * Update B and D from C.
        *
        * @param negccov Negative covariance factor.
        */
        void updateBD(TValue negccov)
        {
            if(ccov1 + ccovmu + negccov > 0 &&
                    (std::fmod(iterations, 1. / (ccov1 + ccovmu + negccov) / dimension / 10.) < 1.))
            {
                // to achieve O(N^2)
                C = triu(C, 0) + (triu(C, 1).transpose());
                // enforce symmetry to prevent complex numbers
                Eigen::SelfAdjointEigenSolver<TMatrix> eig(C);
                B = eig.eigenvectors(); // eigen decomposition, B==normalized eigenvectors
                diagD = eig.eigenvalues();

                if(min(diagD) <= 0)
                {
                    for(int i = 0; i < dimension; i++)
                    {
                        if(diagD(i, 0) < 0)
                        {
                            diagD(i, 0) =  0.;
                        }
                    }

                    TValue tfac = max(diagD) / 1e14;
                    C = C + eye(dimension, dimension) * tfac;
                    diagD += ones(dimension, 1) * tfac;
                }

                if(max(diagD) > 1e14 * min(diagD))
                {
                    TValue tfac = max(diagD) / 1e14 - min(diagD);
                    C += eye(dimension, dimension) * tfac;
                    diagD += ones(dimension, 1) * tfac;
                }

                diagC = diag(C);
                diagD = sqrt(diagD); // D contains standard deviations now
                BD = times(B, repmat(diagD.transpose(), dimension, 1)); // O(n^2)
            }
        }

        /**
        * Pushes the current best fitness value in a history queue, maintaining its length.
        *
        * @param vals History queue.
        * @param val Current best fitness value.
        */
        static void push(std::deque<TValue>& vals, TValue val)
        {
            vals.push_back(val);
            vals.pop_front();
        }

        /**
        * Sorts fitness values (lowest value first).
        *
        * v  Array of values to be sorted.
        * returns a sorted array of indices pointing into doubles.
        */
        static TIntVector sortedIndices(const TVector& v)
        {
            int index = 0;
            TIntVector sorted(v.size());
            std::generate(sorted.data(), sorted.data() + v.size(), [&]()
            {
                return index++;
            });
            std::sort(sorted.data(), sorted.data() + v.size(), [&](int a, int b)
            {
                return v[a] <= v[b];
            });
            return sorted;
        }

        /**
        * Normalizes fitness values to the range [0,1]. Adds a penalty to the
        * fitness value if out of range. The penalty is adjusted by calling
        * setValueRange().
        */
        class FitnessFunction
        {
        private:

            /** Pointer to the outer class */
            CMAESOptimizer<TValue>* outer;

            /** Determines the penalty for boundary violations */
            TValue valueRange;
            /**
            * Flag indicating whether the objective variables are forced into their
            * bounds if defined
            */
            bool isRepairMode;

        public:

            /** Simple constructor.
            */
            FitnessFunction(CMAESOptimizer<TValue>* outer_)
                : outer(outer_), valueRange(1.0), isRepairMode(true)
            {
            }

            /**
            * x  Original objective variables.
            * returns the normalized objective variables.
            */
            TVector encode(const TVector& x)
            {
                if(outer->lowerBoundaries.size() == 0)
                {
                    return x;
                }

                TVector res(x.size());

                for(int i = 0; i < x.size(); i++)
                {
                    TValue diff = outer->upperBoundaries[i] - outer->lowerBoundaries[i];
                    res[i] = (x[i] - outer->lowerBoundaries[i]) / diff;
                }

                return res;
            }

            /**
            * x  Normalized objective variables.
            * returns the original objective variables.
            */
            TVector decode(const TVector& x)
            {
                if(outer->lowerBoundaries.size() == 0)
                {
                    return x;
                }

                TVector res(x.size());

                for(int i = 0; i < x.size(); i++)
                {
                    TValue diff = outer->upperBoundaries[i] - outer->lowerBoundaries[i];
                    res[i] = diff * x[i] + outer->lowerBoundaries[i];
                }

                return res;
            }

            /**
            * point  Normalized objective variables.
            * returns the objective value + penalty for violated bounds.
            */
            TValue value(const TVector& point)
            {
                if(outer->lowerBoundaries.size() != 0 && isRepairMode)
                {
                    TVector repaired = repair(point);
                    return outer->computeObjectiveValue(decode(repaired)) + penalty(point, repaired);
                }
                else
                {
                    return outer->computeObjectiveValue(decode(point));
                }
            }

            /**
            * x  Normalized objective variables.
            * returns true if in bounds.
            */
            bool isFeasible(const TVector& x)
            {
                if(outer->lowerBoundaries.size() == 0)
                {
                    return true;
                }

                for(int i = 0; i < x.size(); i++)
                {
                    if(x[i] < 0)
                    {
                        return false;
                    }

                    if(x[i] > 1.0)
                    {
                        return false;
                    }
                }

                return true;
            }

            /**
            * valueRange  Adjusts the penalty computation.
            */
            void setValueRange(TValue valueRange_)
            {
                valueRange = valueRange_;
            }

            /**
            * x  Normalized objective variables.
            * returns the repaired objective variables - all in bounds.
            */
            TVector repair(const TVector& x)
            {
                TVector repaired(x.size());

                for(int i = 0; i < x.size(); i++)
                {
                    if(x[i] < 0)
                    {
                        repaired[i] = 0;
                    }
                    else if(x[i] > 1.0)
                    {
                        repaired[i] = 1.0;
                    }
                    else
                    {
                        repaired[i] = x[i];
                    }
                }

                return repaired;
            }

            /**
            * x         Normalized objective variables.
            * repaired  Repaired objective variables.
            * returns Penalty value according to the violation of the bounds.
            */
            TValue penalty(const TVector& x, const TVector& repaired)
            {
                TValue penalty = 0;

                for(int i = 0; i < x.size(); i++)
                {
                    TValue diff = std::abs(x[i] - repaired[i]);
                    penalty += diff * valueRange;
                }

                return penalty;
            }
        };

        // -----Matrix utility functions similar to the Matlab build in functions------

        /**
        * m  Input matrix
        * returns Matrix representing the element-wise logarithm of m.
        */
        static TMatrix log(const TMatrix& m)
        {
            return m.array().log();
        }

        /**
        * m  Input matrix
        * returns Matrix representing the element-wise square root of m.
        */
        static TMatrix sqrt(const TMatrix& m)
        {
            return m.array().sqrt();
        }

        /**
        * m  Input matrix
        * returns Matrix representing the element-wise square (^2) of m.
        */
        static TMatrix square(const TMatrix& m)
        {
            return m.array().square();
        }

        /**
        * m  Input matrix 1.
        * n  Input matrix 2.
        * returns the matrix where the elements of m and n are element-wise multiplied.
        */
        static TMatrix times(const TMatrix& m, const TMatrix& n)
        {
            return m.array() * n.array();
        }

        /**
        * m  Input matrix 1.
        * n  Input matrix 2.
        * returns Matrix where the elements of m and n are element-wise divided.
        */
        static TMatrix divide(const TMatrix& m, const TMatrix& n)
        {
            return m.array() / n.array();
        }

        /**
        * m    Input matrix.
        * cols Columns to select.
        * returns Matrix representing the selected columns.
        */
        static TMatrix selectColumns(const TMatrix& m, const TIntVector& cols)
        {
            TMatrix d(m.rows(), cols.size());

            for(int r = 0; r < m.rows(); r++)
            {
                for(int c = 0; c < cols.size(); c++)
                {
                    d(r, c) = m(r, cols[c]);
                }
            }

            return d;
        }

        /**
        * m  Input matrix.
        * k  Diagonal position.
        * returns Upper triangular part of matrix.
        */
        static TMatrix triu(const TMatrix& m, int k)
        {
            TMatrix d(m.rows(), m.cols());

            for(int r = 0; r < m.rows(); r++)
            {
                for(int c = 0; c < m.cols(); c++)
                {
                    d(r, c) = r <= c - k ? m(r, c) : 0;
                }
            }

            return d;
        }

        /**
        * m  Input matrix.
        * returns Row matrix representing the sums of the rows.
        */
        static TMatrix sumRows(const TMatrix& m)
        {
            return m.colwise().sum();
        }

        /**
        * m  Input matrix.
        * returns the diagonal n-by-n matrix if m is a column matrix or the column
        * matrix representing the diagonal if m is a n-by-n matrix.
        */
        static TMatrix diag(const TMatrix& m)
        {
            if(m.cols() == 1)
            {
                return m.col(0).asDiagonal();
            }
            else
            {
                return m.diagonal();
            }
        }

        /**
        * Copies a column from m1 to m2.
        *
        * m1   Source matrix 1.
        * col1 Source column.
        * m2   Target matrix.
        * col2 Target column.
        */
        static void copyColumn(const TMatrix& m1, int col1, TMatrix& m2, int col2)
        {
            m2.col(col2) = m1.col(col1);
        }

        /**
        * n  Number of rows.
        * m  Number of columns.
        * returns n-by-m matrix filled with 1.
        */
        static TMatrix ones(int n, int m)
        {
            return TMatrix::Ones(n, m);
        }

        /**
        * n  Number of rows.
        * m  Number of columns.
        * returns n-by-m matrix of 0.0-values, diagonal has values 1.0.
        */
        static TMatrix eye(int n, int m)
        {
            return TMatrix::Identity(n, m);
        }

        /**
        * n Number of rows.
        * m Number of columns.
        * returns n-by-m matrix of 0.0-values.
        */
        static TMatrix zeros(int n, int m)
        {
            return TMatrix::Zero(n, m);
        }

        /**
        * mat Input matrix.
        * n   Number of row replicates.
        * m   Number of column replicates.
        * returns a matrix which replicates the input matrix in both directions.
        */
        static TMatrix repmat(const TMatrix& mat, int n, int m)
        {
            int rd = mat.rows();
            int cd = mat.cols();
            TMatrix d(n * rd, m * cd);

            for(int r = 0; r < n * rd; r++)
            {
                for(int c = 0; c < m * cd; c++)
                {
                    d(r, c) = mat(r % rd, c % cd);
                }
            }

            return d;
        }

        /**
        * start Start value.
        * end   End value.
        * step  Step size.
        * returns a sequence as column vector.
        */
        static TMatrix sequence(TValue start, TValue end, TValue step)
        {
            int size = (int)((end - start) / step + 1);
            TMatrix d(size, 1);
            TValue value = start;

            for(int r = 0; r < size; r++)
            {
                d(r, 0) = value;
                value += step;
            }

            return d;
        }

        /**
        * m  Input matrix.
        * returns the minimum of the matrix element values.
        */
        static TValue min(const TMatrix& m)
        {
            return m.minCoeff();
        }

        /**
        * m  Input array.
        * returns the maximum of the array values.
        */
        static TValue max(const TVector& m)
        {
            return m.maxCoeff();
        }

        /**
        * m  Input array.
        * returns the minimum of the array values.
        */
        static TValue min(const TVector& m)
        {
            return m.minCoeff();
        }

        /**
        * @param indices Input index array.
        * @return the inverse of the mapping defined by indices.
        */
        static TIntVector inverse(const TIntVector& indices)
        {
            TIntVector inverse(indices.size());

            for(int i = 0; i < indices.size(); i++)
            {
                inverse[indices[i]] = i;
            }

            return inverse;
        }

        /**
        * indices vInput index array.
        * returns the indices in inverse order (last is first).
        */
        static TIntVector reverse(const TIntVector& indices)
        {
            TIntVector reverse(indices.size());

            for(int i = 0; i < indices.size(); i++)
            {
                reverse[i] = indices[indices.size() - i - 1];
            }

            return reverse;
        }

        /**
        * size  Length of random array.
        * returns an array of Gaussian random numbers.
        */
        TVector randn(int size)
        {
            TVector randn(size);

            for(int i = 0; i < size; i++)
            {
                randn[i] = normal(random);
            }

            return randn;
        }

        /**
        * size    Number of rows.
        * popSize Population size.
        * returns a 2-dimensional matrix of Gaussian random numbers.
        */
        TMatrix randn1(int size, int popSize)
        {
            TMatrix d(size, popSize);

            for(int r = 0; r < size; r++)
            {
                for(int c = 0; c < popSize; c++)
                {
                    d(r, c) = normal(random);
                }
            }

            return d;
        }
    };

    namespace Detail
    {
        template <typename TValue>
        struct Default
        {
            static bool Converged(int numIterationsSoFar,
                                  const std::pair<Eigen::Matrix<TValue, Eigen::Dynamic, 1>, TValue>& currentSolution,
                                  const std::pair<Eigen::Matrix<TValue, Eigen::Dynamic, 1>, TValue>& previousSolution)
            {
                return false;
            }
        };
    }

    /*
    * Minimize the provided function via CMA-ES.
    *
    * lambda           Population size - this is the main parameter to tweak;
    *                  a larger population size leads to more global optimization.
    *                  You can set lambda to a value <= 0 to let the implementation
    *                  choose a reasonable default.
    *
    * maxEvaluations   Terminate after at most 'maxEvaluations' function evaluations.
    *
    * x                Initial guess of the solution. The dimensionality of the vector
    *                  must match that of the optimization problem.
    *                  After optimization, the solution is provided in the vector.
    *
    * lowerBoundaries  Lower bounds on the solution variables (one per variable)
    * upperBoundaries  Upper bounds on the solution variables (one per variable)
    *
    * converged        Custom callback that allows to check for convergence of the problem
    *                  in terms of the number of iterations of CMA-ES and the current
    *                  optimum versus the previous optimum.
    */
    template<typename TValue, int lambda>
    TValue
    CMAESMinimize(const std::function<TValue(const Eigen::Matrix<TValue, Eigen::Dynamic, 1>&)>& objective,
                  Eigen::Matrix<TValue, Eigen::Dynamic, 1>& x,
                  int maxEvaluations,
                  const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& lowerBoundaries = Eigen::Matrix<TValue, Eigen::Dynamic, 1>(),
                  const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& upperBoundaries = Eigen::Matrix<TValue, Eigen::Dynamic, 1>(),
                  const std::function < bool (int,
                          const std::pair<Eigen::Matrix<TValue, Eigen::Dynamic, 1>, TValue>&,
                          const std::pair<Eigen::Matrix<TValue, Eigen::Dynamic, 1>, TValue>&) > & converged
                  = Detail::Default<TValue>::Converged)
    {
        CMAESOptimizer<TValue> optimizer(lambda, lowerBoundaries, upperBoundaries, maxEvaluations, converged);
        auto solution = optimizer.doOptimize(objective, x);
        x = solution.first;
        return solution.second;
    }
}

#endif // _H_MINIMIZATION_H_
