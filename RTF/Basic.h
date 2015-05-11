// File:   Basic.h
// Author: t-jejan
//
// Exposes a basic object-oriented interface to training of regression tree fields.
// Please see the interface definition of the RTF class for details.
//
#ifndef _H_BASIC_H_
#define _H_BASIC_H_

#include <string>
#include <random>

#include "Types.h"
#include "Classify.h"
#include "Monitor.h"
#include "Learning.h"
#include "Serialization.h"
#include "LinearOperator.h"


namespace Basic
{

    namespace Detail
    {
        template<typename TWrapped>
        class DatasetAdapter
        {
        public:
            typedef typename TWrapped::UnaryGroundLabel     UnaryGroundLabel;     // INTERFACE: the type to be used for unary groundtruth labels
            typedef typename TWrapped::PairwiseGroundLabel  PairwiseGroundLabel;  // INTERFACE: the type to be used for pairwise groundtruth labels
            typedef typename TWrapped::InputLabel           InputLabel;           // INTERFACE: the label type of input images

        private:
            const TWrapped&                                 original;

            mutable std::vector<VecRef<Vector2D<int>>>      variableSubsamples;
            mutable std::mt19937                            mt;
            mutable std::uniform_int_distribution<int>      dpos;

            double                                          pixelFraction;

        public:

            DatasetAdapter(const TWrapped& original_,
                           double pixelFraction_ = 0.5)
                : original(original_), variableSubsamples(original.GetImageCount()), pixelFraction(pixelFraction_)
            {
                ResampleVariables();
            }

            // INTERFACE: returns the number of images in the dataset
            size_t GetImageCount() const
            {
                return original.GetImageCount();
            }

            // INTERFACE: returns the idx'th ground truth image
            ImageRefC<UnaryGroundLabel> GetGroundTruthImage(size_t idx) const
            {
                return original.GetGroundTruthImage(idx);
            }

            // INTERFACE: returns the idx'th input image.
            ImageRefC<InputLabel> GetInputImage(size_t idx) const
            {
                return original.GetInputImage(idx);
            }

            // SUBSAMPLING INTERFACE: returns a number of subsampled data points
            // TODO: Right now, this samples with replacement; Breimann suggests sampling w/o replacement.
            const VecRef<Vector2D<int>>& GetSubsampledVariables(size_t idx) const
            {
                assert(idx < variableSubsamples.size());

                if(variableSubsamples[idx].empty())
                {
                    auto groundTruth  = GetGroundTruthImage(idx);
                    const auto cx     = groundTruth.Width(), cy = groundTruth.Height();
                    int numSamples    = static_cast<int>(cx * cy * pixelFraction + .5);

                    for(int s = 0; s < numSamples; ++s)
                        variableSubsamples[idx].push_back(Vector2D<int>(dpos(mt) % cx, dpos(mt) % cy));
                }

                return variableSubsamples[idx];
            }

            // Causes a new subsample of variables to be drawn upon the next invocation of GetSubsampledVariables()
            void ResampleVariables() const
            {
                const size_t ci = GetImageCount();

                for(size_t i = 0; i < ci; ++i)
                    variableSubsamples[i].resize(0);
            }
        };
    }

    template < typename TFeatureSampler,
             typename TDataSampler,
             typename TSplitCritTag           = SquaredResidualsCriterion,
             bool UseBasis                    = false,
             bool UseExplicitThresholdTesting = false,
             typename TPrior                  = NullPrior,
             typename TMonitor                = Monitor::DefaultMonitor,
             int CachingMode                  = WEIGHTS_AND_BASIS_PRECOMPUTED,
             typename TLinearOperatorWeights  = LinearOperator::DefaultWeights<typename TDataSampler::UnaryGroundLabel::ValueType> >
    class RTF
    {
    public:
        typedef Traits < TFeatureSampler,
                Detail::DatasetAdapter<TDataSampler>,
                TSplitCritTag,
                TSplitCritTag,
                TPrior,
                TPrior,
                UseBasis,
                UseExplicitThresholdTesting,
                TMonitor,
                CachingMode,
                TLinearOperatorWeights>	TTraits;

        typedef typename TTraits::ValueType         TValue;

        bool discreteInference;

        static const int LBFGS_M = 64;

    private:
        typename TTraits::UnaryFactorTypeVector                 utypes;
        typename TTraits::PairwiseFactorTypeVector              ptypes;
        typename TTraits::LinearOperatorVector                  ltypes;

        std::vector<Learning::Detail::FactorTypeInfo<TValue>>   uinfos;
        std::vector<Learning::Detail::FactorTypeInfo<TValue>>   pinfos;


        void ReadModel(std::istream& in)
        {
            size_t usize;
            in >> usize;
            uinfos.resize(usize);
            for( size_t u = 0; u < usize; ++u )
                in >> uinfos[u];

            size_t psize;
            in >> psize;
            pinfos.resize(psize);
            for( size_t p = 0; p < psize; ++p )
                in >> pinfos[p];

            Serialization::ReadModel<TTraits>(in, utypes, ptypes, ltypes);
        }

        void WriteModel(std::ostream& out) const
        {
            out << uinfos.size() << std::endl;
            for( size_t u = 0; u < uinfos.size(); ++u )
                out << uinfos[u];

            out << pinfos.size() << std::endl;
            for( size_t p = 0; p < pinfos.size(); ++p )
                out << pinfos[p];

            Serialization::WriteModel<TTraits>(out, utypes, ptypes, ltypes);
        }

    public:

        // Deserialization constructor: Load RTF from the file at the given path
        RTF(const std::string& fname) : discreteInference(false)
        {
            std::cerr << "Reading " << fname << std::endl;
            std::ifstream ifs(fname.c_str());
            if( ! ifs )
                throw std::runtime_error("Could not read input file " + fname);
            ReadModel(ifs);
        }

        // Deserialization constructor: Load RTF from the file at the given path
        RTF(const char* fname) : discreteInference(false)
        {
            std::ifstream ifs(fname);
            if( ! ifs )
                throw std::runtime_error("Could not read input file");
            ReadModel(ifs);
        }

        // Deserialization constructor: Load RTF from provided stream
        RTF(std::istream& in) : discreteInference(false)
        {
            ReadModel(in);
        }

        // Default constructor
        RTF() : discreteInference(false)
        {
        }

        // Adds a unary factor type. The parameters specify the characteristics of the underlying regression
        // tree that is to be trained, as well as regularization of the model parameters.
        typename TTraits::UnaryFactorType&
        AddUnaryFactorType(int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                           TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                           TValue linearRegularizationC = TValue(1e-2), TValue quadraticRegularizationC = TValue(1e-2),
                           TValue purityEpsilon = 0)
        {
            utypes.push_back(Learning::MakeUnaryFactorType<TTraits>(smallestEigenValue, largestEigenValue, -1,
                             linearRegularizationC, quadraticRegularizationC));
            uinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
            return utypes.back();
        }

        void FixUnaries()
        {
            for( size_t u = 0; u < utypes.size(); ++u )
                utypes[u].FixParameters(true);
        }

        void FixPairwise()
        {
            for( size_t p = 0; p < ptypes.size(); ++p )
                ptypes[p].FixParameters(true);
        }

        void ResetWeights()
        {
            for( size_t u = 0; u < utypes.size(); ++u )
                utypes[u].ResetWeights();

            for( size_t p = 0; p < ptypes.size(); ++p )
                ptypes[p].ResetWeights();

            for( size_t l = 0; l < ltypes.size(); ++l )
                ltypes[l].ResetWeights();
        }

        // Similar to the above, but for pairwise factor types. The offsets vector specifies which variables (relative to
        // a given pixel) are covered by the factor.
        typename TTraits::PairwiseFactorType&
        AddPairwiseFactorType(const Vector2D<int>& offsets,
                              int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                              TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                              TValue linearRegularizationC = TValue(1e-2), TValue quadraticRegularizationC = TValue(1e-2),
                              TValue purityEpsilon = 0, int quadraticBasisIndex=-1)
        {
            VecRef<Vector2D<int>> offvec;
            offvec.push_back(Vector2D<int>(0, 0));   // the first variable is always 0,0 by convention
            offvec.push_back(offsets);               // offsets of the second variable, relative to 0,0
            ptypes.push_back(Learning::MakePairwiseFactorType<TTraits>(offvec, smallestEigenValue, largestEigenValue, quadraticBasisIndex,
                             linearRegularizationC, quadraticRegularizationC));
            pinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
            return ptypes.back();
        }

        typename TTraits::LinearOperatorRef
        AddLinearOperator(int type)
        {
            ltypes.push_back(TTraits::LinearOperatorRef::Instantiate(type));
            return ltypes.back();
        }

        template<bool Subsample>
        void Learn(const TDataSampler& traindb,
                   size_t maxNumOptimItPerRound = 50,
                   size_t maxNumOptimItFinal    = 50,
                   TValue finalBreakEps         = 1e-3,
                   TValue subsampleFactor       = 0.3)
        {
            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointly<TTraits, Subsample, LBFGS_M>(utypes, uinfos,
                    ptypes, pinfos,
                    adapter, maxNumOptimItPerRound,
                    maxNumOptimItFinal, finalBreakEps);
        }

        template<typename TLossTag, bool Subsample>
        void LearnDiscriminative(const TDataSampler& traindb,
                                 size_t maxNumOptimItPerRound = 50,
                                 size_t maxNumOptimItFinal    = 50,
                                 TValue finalBreakEps         = 1e-3,
                                 bool stagedTraining          = false,
                                 size_t maxNumItCG            = 10000,
                                 TValue residualTolCG         = 1e-4,
                                 TValue subsampleFactor       = 0.3)
        {
            discreteInference = Loss::Loss<TTraits, TLossTag>::RequiresDiscreteInference();
            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointlyDiscriminative<TTraits, TLossTag, Subsample, LBFGS_M>(utypes, uinfos,
                    ptypes, pinfos, ltypes,
                    adapter, maxNumOptimItPerRound,
                    maxNumOptimItFinal, finalBreakEps,
                    stagedTraining, maxNumItCG, residualTolCG);
        }

        ImageRefC<typename TTraits::UnaryGroundLabel>
        Regress(const ImageRefC<typename TTraits::InputLabel>& input, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            return Classify::Predict<TTraits>(utypes, ptypes, ltypes, input, residualTolCG, maxNumItCG, discreteInference);
        }

        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
                Regress(const TDataSampler& testdb, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());

            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = Regress(testdb.GetInputImage(i), maxNumItCG, residualTolCG);

            return predictions;
        }

        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMicroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction) const
        {
            return Loss::MicroAveraged<TTraits, TLossTag>(testdb, prediction);
        }

        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMacroAveraged(const TDataSampler& testdb, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            return Loss::MacroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                return pred;
            });
        }

        template<typename TLossTag, typename TOp>
        typename TTraits::ValueType
        EvaluateMicroAveraged(const TDataSampler& testdb, size_t maxNumItCG, TValue residualTolCG, const TOp& op) const
        {
            return Loss::MicroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                op(pred, idx);
                return pred;
            });
        }

        template<typename TLossTag, typename TOp>
        typename TTraits::ValueType
        EvaluateMacroAveraged(const TDataSampler& testdb, size_t maxNumItCG, TValue residualTolCG, const TOp& op) const
        {
            return Loss::MacroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                op(pred, idx);
                return pred;
            });
        }

        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMicroAveraged(const TDataSampler& testdb, size_t maxNumItCG, TValue residualTolCG) const
        {
            return Loss::MicroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                return pred;
            });
        }

        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMacroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction) const
        {
            return Loss::MacroAveraged<TTraits, TLossTag>(testdb, prediction);
        }

        template<typename TLossTag>
        typename TTraits::ValueType
        Evaluate(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                 const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
        {
            return Loss::PerImage<TTraits, TLossTag>(ground, prediction);
        }

        void Serialize(const std::string& fname) const
        {
            std::ofstream ofs(fname.c_str());
            WriteModel(ofs);
        }

        void Serialize(std::ostream& out) const
        {
            WriteModel(out);
        }
    };
}

#endif // _H_BASIC_H_
