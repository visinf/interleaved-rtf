#ifndef _H_STACKED_H_
#define _H_STACKED_H_

// File:   Stacked.h
// Author: jermyj
//
// Exposes an object-oriented interface to stacked training of regression tree fields.
//
#include <string>
#include <random>

#include "Types.h"
#include "Classify.h"
#include "Learning.h"
#include "Serialization.h"
#include "LinearOperator.h"

namespace Stacked
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
             int CachingMode                  = WEIGHTS_AND_BASIS_PRECOMPUTED,
             typename TLinearOperatorWeights  = LinearOperator::DefaultWeights<typename TDataSampler::UnaryGroundLabel::ValueType> >
    class RTF
    {
    public:
        typedef Traits < TFeatureSampler,
                Detail::DatasetAdapter<TDataSampler>,
                TSplitCritTag,
                TSplitCritTag,
                NullPrior,
                NullPrior,
                UseBasis,
                UseExplicitThresholdTesting,
                Monitor::DefaultMonitor,
                CachingMode,
                TLinearOperatorWeights>	TTraits;

        typedef typename TTraits::ValueType TValue;

        bool discreteInference;

        static const int LBFGS_M = 64;

    private:
        typedef std::function<typename TTraits::UnaryFactorType()>    UnaryTypeInstantiator;
        typedef std::function<typename TTraits::PairwiseFactorType()> PairwiseTypeInstantiator;
        typedef std::function<typename TTraits::LinearOperatorRef()>  LinearOperatorInstantiator;


        std::vector<typename TTraits::UnaryFactorTypeVector>          utypes;
        std::vector<typename TTraits::PairwiseFactorTypeVector>       ptypes;
        std::vector<typename TTraits::LinearOperatorVector>           ltypes;

        std::vector<Learning::Detail::FactorTypeInfo<TValue>>         uinfos;
        std::vector<UnaryTypeInstantiator>                            ucreat;
        std::vector<Learning::Detail::FactorTypeInfo<TValue>>         pinfos;
        std::vector<PairwiseTypeInstantiator>                         pcreat;
        std::vector<LinearOperatorInstantiator>                       lcreat;

        void AddFactorTypes()
        {
            typename TTraits::UnaryFactorTypeVector ut;

            for(size_t u = 0; u < ucreat.size(); ++u)
                ut.push_back(ucreat[u]());

            utypes.push_back(ut);
            typename TTraits::PairwiseFactorTypeVector pt;

            for(size_t p = 0; p < pcreat.size(); ++p)
                pt.push_back(pcreat[p]());

            ptypes.push_back(pt);

            typename TTraits::LinearOperatorVector lt;

            for( size_t l = 0; l < lcreat.size(); ++l )
                lt.push_back(lcreat[l]());

            ltypes.push_back(lt);
        }

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

            size_t ensembleSize;
            in >> ensembleSize;

            utypes.resize(ensembleSize);
            ptypes.resize(ensembleSize);
            ltypes.resize(ensembleSize);
            for( size_t idx = 0; idx < ensembleSize; ++idx )
                Serialization::ReadModel<TTraits>(in, utypes[idx], ptypes[idx], ltypes[idx]);
        }

        void WriteModel(std::ostream& out) const
        {
            out << uinfos.size() << std::endl;
            for( size_t u = 0; u < uinfos.size(); ++u )
                out << uinfos[u];

            out << pinfos.size() << std::endl;
            for( size_t p = 0; p < pinfos.size(); ++p )
                out << pinfos[p];

            out << EnsembleSize() << std::endl;

            for( size_t idx = 0; idx < EnsembleSize(); ++idx )
                Serialization::WriteModel<TTraits>(out, utypes[idx], ptypes[idx], ltypes[idx]);
        }

        ImageRefC<typename TTraits::UnaryGroundLabel>
        RegressWith(const ImageRefC<typename TTraits::InputLabel>& input, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            return Classify::Predict<TTraits>(utypes[m], ptypes[m], ltypes[m],
                                              input, residualTolCG, (unsigned) maxNumItCG, discreteInference);
        }

        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
                RegressWith(const TDataSampler& testdb, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());
            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = RegressWith(testdb.GetInputImage(i), m, maxNumItCG, residualTolCG);

            return predictions;
        }

        void InitializeDataset(const TDataSampler& ds, size_t maxNumItCG, TValue residualTolCG, int upToLevel=-1) const
        {
            if( upToLevel < 0 )
                upToLevel = static_cast<int>(EnsembleSize())-1;

            // Evaluate any previously trained models on the new data and give the user
            // the chance to store the predictions of these models within the dataset.
            for( size_t m = 0; m <= upToLevel; ++m ) {
                if( m == 0 )
                    ds.InitializeForCascadeLevel(0, VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>() );
                else
                    ds.InitializeForCascadeLevel(m, RegressWith(ds, m-1, maxNumItCG, residualTolCG));
            }
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

        // Number of models in the stacked ensemble
        size_t EnsembleSize() const
        {
            return utypes.size();
        }

        // Adds a unary factor type. The parameters specify the characteristics of the underlying regression
        // tree that is to be trained, as well as regularization of the model parameters.
        void AddUnaryFactorType(int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                                TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                                TValue purityEpsilon = 0)
        {
            ucreat.push_back([ = ]()
            {
                return Learning::MakeUnaryFactorType<TTraits>(smallestEigenValue, largestEigenValue);
            });
            uinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
        }


        // Similar to the above, but for pairwise factor types. The offsets vector specifies which variables (relative to
        // a given pixel) are covered by the factor.
        void AddPairwiseFactorType(const Vector2D<int>& offsets,
                                   int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                                   TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                                   TValue purityEpsilon = 0, int quadraticBasisIndex=-1)
        {
            VecRef<Vector2D<int>> offvec;
            offvec.push_back(Vector2D<int>(0, 0));   // the first variable is always 0,0 by convention
            offvec.push_back(offsets);               // offsets of the second variable, relative to 0,0
            pcreat.push_back([ = ]()
            {
                return Learning::MakePairwiseFactorType<TTraits>(offvec,
                        smallestEigenValue, largestEigenValue, quadraticBasisIndex);
            });
            pinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
        }

        void AddLinearOperator(int type)
        {
            lcreat.push_back([ = ]()
            {
                return TTraits::LinearOperatorRef::Instantiate(type);
            });
        }


        template<bool Subsample>
        void LearnOneMore(const TDataSampler& traindb,
                          size_t maxNumOptimItPerRound = 50,
                          size_t maxNumOptimItFinal    = 50,
                          TValue finalBreakEps         = 1e-3,
                          TValue subsampleFactor       = 0.3,
                          size_t maxNumItCG            = 10000,
                          TValue residualTolCG         = 1e-4)
        {
            AddFactorTypes();
            InitializeDataset(traindb, maxNumItCG, residualTolCG);

            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointly<TTraits, Subsample, LBFGS_M>(utypes.back(), uinfos,
                    ptypes.back(), pinfos,
                    adapter, maxNumOptimItPerRound,
                    maxNumOptimItFinal, finalBreakEps);
        }

        template<typename TLossTag, bool Subsample>
        void LearnOneMoreDiscriminative(TDataSampler& traindb,
                                        size_t maxNumOptimItPerRound = 50,
                                        size_t maxNumOptimItFinal    = 50,
                                        TValue finalBreakEps         = 1e-3,
                                        bool stagedTraining          = false,
                                        size_t maxNumItCG            = 10000,
                                        TValue residualTolCG         = 1e-4,
                                        TValue subsampleFactor       = 0.3)
        {
            discreteInference = Loss::Loss<TTraits, TLossTag>::RequiresDiscreteInference();

            AddFactorTypes();
            InitializeDataset(traindb, maxNumItCG, residualTolCG);

            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointlyDiscriminative<TTraits, TLossTag, Subsample, LBFGS_M>(utypes.back(), uinfos,
                    ptypes.back(), pinfos, ltypes.back(),
                    adapter, maxNumOptimItPerRound,
                    maxNumOptimItFinal, finalBreakEps,
                    stagedTraining, maxNumItCG, residualTolCG);
        }

        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
                Regress(TDataSampler& testdb, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6, int upToLevel=-1) const
        {
            if( upToLevel < 0 )
                upToLevel = static_cast<int>(EnsembleSize()) - 1;

            InitializeDataset(testdb, maxNumItCG, residualTolCG, upToLevel);

            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());
            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = RegressWith(testdb.GetInputImage(i), upToLevel, maxNumItCG, residualTolCG);

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

#endif // _H_STACKED_H_
