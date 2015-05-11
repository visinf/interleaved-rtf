#ifndef _H_BAGGED_H_
#define _H_BAGGED_H_

// File:   BAGGED.h
// Author: jermyj
//
// Exposes an object-oriented interface to BAGGED training of regression tree fields.
//
#include <string>
#include <random>

#include "Types.h"
#include "Classify.h"
#include "Learning.h"
#include "Serialization.h"
#include "LinearOperator.h"

namespace Bagged
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

            mutable std::vector<size_t>                     imagePermutation;

        public:

            DatasetAdapter(const TWrapped& original_,
                           double pixelFraction_ = 0.5)
                : original(original_), variableSubsamples(original.GetImageCount()), mt(std::rand()),
                  pixelFraction(pixelFraction_), imagePermutation(original.GetImageCount())
            {
                ResampleImages();
                ResampleVariables();
            }

            // INTERFACE: returns the number of images in the dataset
            size_t GetImageCount() const
            {
                return imagePermutation.size();
            }

            // INTERFACE: returns the idx'th ground truth image
            ImageRefC<UnaryGroundLabel> GetGroundTruthImage(size_t idx) const
            {
                return original.GetGroundTruthImage(imagePermutation[idx]);
            }

            // INTERFACE: returns the idx'th input image.
            ImageRefC<InputLabel> GetInputImage(size_t idx) const
            {
                return original.GetInputImage(imagePermutation[idx]);
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

            // Re-samples the images of the original dataset with replacement
            void ResampleImages() const
            {
                const auto ci = GetImageCount();
                for( size_t idx = 0; idx < ci; ++idx )
                    imagePermutation[idx] = dpos(mt) % ci;
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

        bool discreteInference = false;

        static const int LBFGS_M = 64;

    private:
        typedef std::function<typename TTraits::UnaryFactorType()>            UnaryTypeInstantiator;
        typedef std::function<typename TTraits::PairwiseFactorType()>         PairwiseTypeInstantiator;
        typedef std::function<typename TTraits::LinearOperatorRef()>          LinearOperatorInstantiator;

        std::vector<std::vector<typename TTraits::UnaryFactorTypeVector>>     utypes;
        std::vector<std::vector<typename TTraits::PairwiseFactorTypeVector>>  ptypes;
        std::vector<std::vector<typename TTraits::LinearOperatorVector>>      ltypes;

        std::vector<Learning::Detail::FactorTypeInfo<TValue>>                 uinfos;
        std::vector<UnaryTypeInstantiator>                                    ucreat;
        std::vector<Learning::Detail::FactorTypeInfo<TValue>>                 pinfos;
        std::vector<PairwiseTypeInstantiator>                                 pcreat;
        std::vector<LinearOperatorInstantiator>                               lcreat;

        size_t                                                                bsize;

        void AddFactorTypes()
        {
            utypes.resize( utypes.size() + 1 );
            ptypes.resize( ptypes.size() + 1 );
            ltypes.resize( ltypes.size() + 1 );

            for( size_t b = 0; b < BagSize(); ++b )
            {
                typename TTraits::UnaryFactorTypeVector ut;
                for(size_t u = 0; u < ucreat.size(); ++u)
                    ut.push_back(ucreat[u]());
                utypes.back().push_back(ut);

                typename TTraits::PairwiseFactorTypeVector pt;
                for(size_t p = 0; p < pcreat.size(); ++p)
                    pt.push_back(pcreat[p]());
                ptypes.back().push_back(pt);

                typename TTraits::LinearOperatorVector lt;
                for( size_t l = 0; l < lcreat.size(); ++l )
                    lt.push_back(lcreat[l]());
                ltypes.back().push_back(lt);
            }
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
            size_t bagSize;
            in >> bagSize;

            utypes.resize(ensembleSize);
            ptypes.resize(ensembleSize);
            ltypes.resize(ensembleSize);
            for( size_t idx = 0; idx < ensembleSize; ++idx )
            {
                utypes[idx].resize(bagSize);
                ptypes[idx].resize(bagSize);
                ltypes[idx].resize(bagSize);

                for( size_t b = 0; b < bagSize; ++b )
                {
                    Serialization::ReadModel<TTraits>(in, utypes[idx][b], ptypes[idx][b], ltypes[idx][b]);
                }
            }
            bsize = BagSize;
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
            out << BagSize() << std::endl;

            for( size_t idx = 0; idx < EnsembleSize(); ++idx )
            {
                for( size_t b = 0; b < BagSize(); ++b )
                {
                    Serialization::WriteModel<TTraits>(out, utypes[idx][b], ptypes[idx][b], ltypes[idx][b]);
                }
            }
        }

        ImageRefC<typename TTraits::UnaryGroundLabel>
        RegressWith(const ImageRefC<typename TTraits::InputLabel>& input, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            const int cx = input.Width(), cy = input.Height();
            ImageRef<typename TTraits::UnaryGroundLabel> pred(cx, cy);
            pred.Clear();
            double pred_factor = 1.0 / static_cast<double>(BagSize());	// uniform average

            for(size_t b = 0; b < BagSize(); ++b)
            {
                // Get prediction of current model
                auto cur_pred = Classify::Predict<TTraits>(utypes[m][b], ptypes[m][b], ltypes[m][b],
                                input, residualTolCG, (unsigned) maxNumItCG, discreteInference);

                // Per-pixel average
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        const auto cc = TTraits::UnaryGroundLabel::Size;
                        typename TTraits::UnaryGroundLabel pix_pred = cur_pred(x, y);

                        for(size_t c = 0; c < cc; ++c)
                        {
                            pred(x, y)[c] += pred_factor * pix_pred[c];
                        }
                    }
                }
            }

            return pred;
        }

        std::pair<ImageRefC<typename TTraits::UnaryGroundLabel>, ImageRefC<typename TTraits::UnaryGroundLabel>>
                RegressWithVariance(const ImageRefC<typename TTraits::InputLabel>& input, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            const int cx = input.Width(), cy = input.Height(), cc = TTraits::UnaryGroundLabel::Size;
            ImageRef<typename TTraits::UnaryGroundLabel> pred(cx, cy), var(cx, cy);
            pred.Clear();
            var.Clear();
            double pred_factor = 1.0 / static_cast<double>(BagSize());	// uniform average


            for(size_t b = 0; b < BagSize(); ++b)
            {
                // Get prediction of current model
                auto cur_pred = Classify::Predict<TTraits>(utypes[m][b], ptypes[m][b], ltypes[m][b],
                                input, residualTolCG, (unsigned) maxNumItCG, discreteInference);

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        const auto pix_pred = cur_pred(x, y);

                        for(size_t c = 0; c < cc; ++c)
                        {
                            pred(x, y)[c] += pred_factor * pix_pred[c];
                            var (x, y)[c] += pix_pred[c] * pix_pred[c];
                        }
                    }
                }
            }

            #pragma omp parallel for
            for(int y = 0; y < cy; ++y)
            {
                for(int x = 0; x < cx; ++x)
                {
                    const auto pix_pred = pred(x, y);

                    for(size_t c = 0; c < cc; ++c)
                        var(x, y)[c] -= (pix_pred[c]*pix_pred[c])/pred_factor;
                }
            }

            return std::make_pair(pred, var);
        }

        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
        RegressWith(const TDataSampler& testdb, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());
            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = RegressWith(testdb.GetInputImage(i), m, maxNumItCG, residualTolCG);

            return predictions;
        }

        VecCRef<std::pair<ImageRefC<typename TTraits::UnaryGroundLabel>, ImageRefC<typename TTraits::UnaryGroundLabel>>>
        RegressWithVariance(const TDataSampler& testdb, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            VecRef<std::pair<ImageRefC<typename TTraits::UnaryGroundLabel>, ImageRefC<typename TTraits::UnaryGroundLabel>>> predictions(testdb.GetImageCount());
            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = RegressWithVariance(testdb.GetInputImage(i), m, maxNumItCG, residualTolCG);

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
                    ds.InitializeForCascadeLevel(0, VecCRef<std::pair<ImageRefC<typename TTraits::UnaryGroundLabel>, ImageRefC<typename TTraits::UnaryGroundLabel>>>());
                else
                    ds.InitializeForCascadeLevel(m, RegressWithVariance(ds, m-1, maxNumItCG, residualTolCG));
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
        RTF(size_t bagSize) : bsize(bagSize), discreteInference(false)
        {
        }

        // Number of models in the bagged ensemble
        size_t EnsembleSize() const
        {
            return utypes.size();
        }

        size_t BagSize() const
        {
            return bsize;
        }

        // Adds a unary factor type. The parameters specify the characteristics of the underlying regression
        // tree that is to be trained, as well as regularization of the model parameters.
        void AddUnaryFactorType(int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                                TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                                int quadraticBasisIndex = -1,
                                TValue purityEpsilon = 0)
        {
            ucreat.push_back([ = ]()
            {
                return Learning::MakeUnaryFactorType<TTraits>(smallestEigenValue, largestEigenValue, quadraticBasisIndex);
            });
            uinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
        }

        Learning::Detail::FactorTypeInfo<TValue>& GetUnaryInfo(size_t uindex = 0)
        {
            return uinfos[uindex];
        }

        Learning::Detail::FactorTypeInfo<TValue>& GetPairwiseInfo(size_t pindex)
        {
            return pinfos[pindex];
        }

        size_t NumPairwise() const
        {
            return pinfos.size();
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

            for( size_t b = 0; b < BagSize(); ++b )
            {
                TTraits::Monitor::Report("Training model with bag index %u\n", b);
                Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
                Learning::LearnTreesAndWeightsJointly<TTraits, Subsample, LBFGS_M>(utypes.back()[b], uinfos,
                        ptypes.back()[b], pinfos,
                        adapter, maxNumOptimItPerRound,
                        maxNumOptimItFinal, finalBreakEps);
            }
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

            for( size_t b = 0; b < BagSize(); ++b )
            {
                TTraits::Monitor::Report("Training model with bag index %u\n", b);
                Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
                Learning::LearnTreesAndWeightsJointlyDiscriminative<TTraits, TLossTag, Subsample, LBFGS_M>(utypes.back()[b], uinfos,
                        ptypes.back()[b], pinfos, ltypes.back()[b],
                        adapter, maxNumOptimItPerRound,
                        maxNumOptimItFinal, finalBreakEps,
                        stagedTraining, maxNumItCG, residualTolCG);
            }
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

#endif // _H_BAGGED_H_
