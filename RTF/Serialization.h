// File:   Serialization.h
// Author: t-jejan
//
// Implements functionality that allows to dump RTF models to disk and to read them
// in again at a later point in time.
//
// If you want your model to be serialiazable, your Feature class must implement the
// stream operators
//
//   std::ostream & operator<<(std::ostream& os, const Feature& feat);
//
// and
//
//   std::istream& operator>>(std::istream& is, Feature& feat);
//
// in a fully self-contained manner. It is recommended to use a plain text representation
// of your features.
//
// You can then use the following API functions:
//
// - WriteModel()
//     Writes a plain text representation of your RTF model to the given output stream.
//     This comprises the structure of the underlying regression tree of each factor type,
//     including the features at the inner nodes and the weights at the leaves.
//     You will still have to set up the model traits at compile time, however.
// - ReadModel()
//     Recovers the contents of a factor type from the given input stream. The underlying
//     regression trees will be instantiated and populated with the feature instances
//     at the nodes and the weights instances at the leaves.
//
// - WriteTree()
//     Writes a plain text representation of a given tree to the given output stream.
// - ReadTree()
//     Recovers the contents of a regression tree from the given input stream.
//
// See below for details.
//
#ifndef _H_SERIALIZATION_H_
#define _H_SERIALIZATION_H_

#include <string>
#include <iostream>

namespace Compute
{
    template<typename TValue, size_t VarDim, size_t BasisDim>
    std::ostream& operator<< (std::ostream& out, const Weights<TValue, VarDim, BasisDim>& weights)
    {
        out << weights.Wl << std::endl;
        out << weights.Wq << std::endl;
        return out;
    }

    template<typename TValue, size_t VarDim, size_t BasisDim>
    std::istream& operator>>(std::istream& in, Weights<TValue, VarDim, BasisDim>& weights)
    {
        for(size_t row = 0; row < VarDim; ++row)
            for(size_t column = 0; column < BasisDim; ++column)
                in >> weights.Wl(row, column);

        for(size_t row = 0; row < VarDim; ++row)
            for(size_t column = 0; column < VarDim; ++column)
                in >> weights.Wq(row, column);

        return in;
    }

    template<typename TFeature, typename TLabel, typename TPrior, typename TBasis>
    std::ostream& operator<<(std::ostream& out, const FactorTypeBase<TFeature, TLabel, TPrior, TBasis>& type)
    {
        out << type.smallestEigenvalue << " " << type.largestEigenvalue << std::endl;
        out << type.linearRegularizationC << " " << type.quadraticRegularizationC << std::endl;

        if(type.offsets.size() == 2)
            for(size_t v = 0; v < 2; ++v)
                out << type.offsets[v].x << " " << type.offsets[v].y << std::endl;

        const auto begin = type.tree.begin_breadth_first(), end = type.tree.end_breadth_first();

        for(auto it = begin; it != end; ++it)
        {
            out << it.number_of_children() << std::endl << std::endl;
            out << it->feature << std::endl;
            out << it->data << std::endl;
        }

        out << std::endl;
        return out;
    }

    template<typename TFeature, typename TLabel, typename TPrior, typename TBasis>
    std::istream& operator>>(std::istream& in, FactorTypeBase<TFeature, TLabel, TPrior, TBasis>& type)
    {
        typedef Weights<typename TLabel::ValueType, TLabel::Size, TBasis::Size> TWeights;
        type.tree.set_head(NodeData<TFeature, TWeights>(TFeature(), TWeights()));
        const auto begin = type.tree.begin_breadth_first(), end = type.tree.end_breadth_first();

        for(auto it = begin; it != end; ++it)
        {
            size_t numChildren;
            in >> numChildren;
            in >> it->feature;
            in >> it->data;

            for(size_t c = 0; c < numChildren; ++c)
                type.tree.append_child(it, NodeData<TFeature, TWeights>(TFeature(), TWeights()));
        }

        return in;
    }
}

namespace Training
{
    template<typename TLabel>
    std::ostream& operator<< (std::ostream& out, const RegressionTreeNode<TLabel>& node)
    {
        out << node.average << std::endl;
        out << node.numDataPoints << std::endl;
        out << node.trained << std::endl;
        return out;
    }

    template<typename TLabel>
    std::istream& operator>>(std::istream& in, RegressionTreeNode<TLabel>& node)
    {
        for(size_t c = 0; c < TLabel::Size; ++c)
            in >> node.average[c];

        in >> node.numDataPoints;
        in >> node.trained;
        return in;
    }
}

namespace LinearOperator
{
    template<typename TFeature, typename TLabel, typename TWeights>
    std::ostream& operator<< (std::ostream& out, const OperatorRef<TFeature, TLabel, TWeights>& op)
    {
        op.Serialize(out);
        return out;
    }

    template<typename TFeature, typename TLabel, typename TWeights>
    std::istream& operator>>(std::istream& in, OperatorRef<TFeature, TLabel, TWeights>& op)
    {
        op.Deserialize(in);
        return in;
    }
}

namespace Learning
{
    namespace Detail
    {
        template<typename TValue>
        std::ostream& operator<< (std::ostream& out, const FactorTypeInfo<TValue>& info)
        {
            out << info.purityEpsilon << " " << info.nFeatureCount << " " << info.nDepthLevels << " " << info.nMinDataPointsForSplitConsideration << std::endl;
            return out;
        }

        template<typename TValue>
        std::istream& operator>> (std::istream& in, FactorTypeInfo<TValue>& info)
        {
            in >> info.purityEpsilon >> info.nFeatureCount >> info.nDepthLevels >> info.nMinDataPointsForSplitConsideration;
            return in;
        }
    }
}

namespace Serialization
{
    // Dumps the given model to output stream 'out'.
    template < typename TTraits >
    std::ostream& WriteModel(std::ostream& out,
                             const typename TTraits::UnaryFactorTypeVector& Us,
                             const typename TTraits::PairwiseFactorTypeVector& Ps,
                             const typename TTraits::LinearOperatorVector& Ls)
    {
        out << Us.size() << std::endl << std::endl;
        std::for_each(Us.begin(), Us.end(), [&](const typename TTraits::UnaryFactorType & U)
        {
            out << U;
            out << U.GetQuadraticBasisIndex() << std::endl;
        });
        out << Ps.size() << std::endl << std::endl;
        std::for_each(Ps.begin(), Ps.end(), [&](const typename TTraits::PairwiseFactorType & P)
        {
            out << P;
            out << P.GetQuadraticBasisIndex() << std::endl;
        });
        out << Ls.size() << std::endl << std::endl;
        std::for_each(Ls.begin(), Ls.end(), [&](const typename TTraits::LinearOperatorRef& L)
        {
            out << L.Type() << std::endl;
            out << L;
        });
        return out;
    }

    // Same as above, but directly writes to the specified path instead.
    template < typename TTraits >
    void WriteModel(const std::string& path,
                    const typename TTraits::UnaryFactorTypeVector& Us,
                    const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::LinearOperatorVector& Ls)
    {
        std::ofstream out(path);
        WriteModel<TTraits>(out, Us, Ps, Ls);
    }

    // Recovers a factor type from the given input stream. You must pass in
    // empty vectors for the unary and the pairwise factor types, which will
    // then be populated by the function.
    template < typename TTraits >
    std::istream& ReadModel(std::istream& in,
                            typename TTraits::UnaryFactorTypeVector& Us,
                            typename TTraits::PairwiseFactorTypeVector& Ps,
                            typename TTraits::LinearOperatorVector& Ls)
    {
        size_t numUs;
        in >> numUs;
        std::cerr << numUs << " unary factor types" << std::endl;
        Us.clear();

        for(size_t u = 0; u < numUs; ++u)
        {
            typename TTraits::ValueType smallestEigenvalue, largestEigenvalue, linearRegularizationC, quadraticRegularizationC;
            in >> smallestEigenvalue >> largestEigenvalue;
            in >> linearRegularizationC >> quadraticRegularizationC;

            Us.push_back(Learning::Detail::MakeUnaryFactorType<TTraits>(smallestEigenvalue, largestEigenvalue, linearRegularizationC, quadraticRegularizationC));
            in >> Us.back();
            int quadraticBasisIndex;
            in >> quadraticBasisIndex;
            std::cerr << "quadratic basis index: " << quadraticBasisIndex << std::endl;
            Us.back().SetQuadraticBasisIndex(quadraticBasisIndex);
        }

        size_t numPs;
        in >> numPs;
        Ps.clear();
        std::cerr << numPs << " pairwise factor types" << std::endl;

        for(size_t p = 0; p < numPs; ++p)
        {
            typename TTraits::ValueType smallestEigenvalue, largestEigenvalue, linearRegularizationC, quadraticRegularizationC;
            in >> smallestEigenvalue >> largestEigenvalue;
            in >> linearRegularizationC >> quadraticRegularizationC;

            VecRef<Vector2D<int>> offsets(2);

            for(size_t v = 0; v < 2; ++v)
                in >> offsets[v].x >> offsets[v].y;

            Ps.push_back(Learning::Detail::MakePairwiseFactorType<TTraits>(offsets, smallestEigenvalue, largestEigenvalue, linearRegularizationC, quadraticRegularizationC));
            in >> Ps.back();
            int quadraticBasisIndex;
            in >> quadraticBasisIndex;
            std::cerr << "quadratic basis index: " << quadraticBasisIndex << std::endl;
            Ps.back().SetQuadraticBasisIndex(quadraticBasisIndex);
        }

        size_t numLs;
        in >> numLs;
        Ls.clear();

        std::cerr << numLs << " linear operators" << std::endl;
        for( size_t l = 0; l < numLs; ++l )
        {
            int type;
            in >> type;
            Ls.push_back(TTraits::LinearOperatorRef::Instantiate(type));
            in >> Ls.back();
        }

        return in;
    }

    // Same as the above, but reads from the given path instead.
    template < typename TTraits >
    void ReadModel(const std::string& path,
                   typename TTraits::UnaryFactorTypeVector& Us,
                   typename TTraits::PairwiseFactorTypeVector& Ps,
                   typename TTraits::LinearOperatorVector& Ls = typename TTraits::LinearOperatorVector())
    {
        std::cerr << "Reading " << path << std::endl;
        std::ifstream in(path);
        ReadModel<TTraits>(in, Us, Ps, Ls);
    }

    // Recovers a plain regression tree from the provided input streem
    template < typename TFeature, typename TData, typename TAllocator >
    std::istream& ReadTree(std::istream& in,
                           TreeRef<TFeature, TData, TAllocator> tree)
    {
        tree.set_head(NodeData<TFeature, TData>(TFeature(), TData()));
        const auto begin = tree.begin_breadth_first(), end = tree.end_breadth_first();

        for(auto it = begin; it != end; ++it)
        {
            size_t numChildren;
            in >> numChildren;
            in >> it->feature;
            in >> it->data;

            for(size_t c = 0; c < numChildren; ++c)
                tree.append_child(it, NodeData<TFeature, TData>(TFeature(), TData()));
        }

        return in;
    }

    // Dumps a plain regression tree to the provided output stream
    template < typename TFeature, typename TData, typename TAllocator >
    std::ostream& WriteTree(std::ostream& out,
                            TreeRef<TFeature, TData, TAllocator> tree)
    {
        const auto begin = tree.begin_breadth_first(), end = tree.end_breadth_first();

        for(auto it = begin; it != end; ++it)
        {
            out << it.number_of_children() << std::endl << std::endl;
            out << it->feature << std::endl;
            out << it->data << std::endl;
        }

        out << std::endl;
        return out;
    }
}

#endif // _H_SERIALIZATION_H_
