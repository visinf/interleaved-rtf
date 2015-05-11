// File:   Boosting.h
// Author: t-jejan
//
// Implements gradient boosting of regression tree fields, along with common loss functions.
// This allows to train an ensemble of RTF classifiers specifically tuned for a specific
// error metric (e.g. MSE, multi-nomial logistic loss).
//
// This module exposes an object-oriented interface. Please see the definition of the BoostedRTF
// class for full details on the available methods.
//
#ifndef _H_BOOSTING_H_
#define _H_BOOSTING_H_

#include <vector>

#include "Types.h"
#include "Loss.h"
#include "Classify.h"
#include "Learning.h"
#include "Minimization.h"
#include "Serialization.h"

namespace Boosting
{

    namespace Detail
    {

        // Wraps a dataset for boosted training. This enables to modify the "response" vector that is
        // learnt by an RTF at the current boosting iteration according to the negative gradient with
        // respect to the prediction of the previous RTF.
        // It also enables sub-sampling of datapoints even for dataset classes that don't support it.
        template<typename TWrapped>
        class PseudoResponseAdapter
        {
        public:
            typedef typename TWrapped::UnaryGroundLabel     UnaryGroundLabel;     // INTERFACE: the type to be used for unary groundtruth labels
            typedef typename TWrapped::PairwiseGroundLabel  PairwiseGroundLabel;  // INTERFACE: the type to be used for pairwise groundtruth labels
            typedef typename TWrapped::InputLabel           InputLabel;           // INTERFACE: the label type of input images

        private:
            const TWrapped&                                 original;
            VecCRef<ImageRef<UnaryGroundLabel>>             response;

            mutable std::vector<VecRef<Vector2D<int>>>      variableSubsamples;
            mutable std::mt19937                            mt;
            mutable std::uniform_int_distribution<int>      dpos;

            double                                          pixelFraction;

        public:

            PseudoResponseAdapter(const TWrapped& original_,
                                  const VecCRef<ImageRef<UnaryGroundLabel>>& response_,
                                  double pixelFraction_ = 0.5)
                : original(original_), response(response_), variableSubsamples(original.GetImageCount()), pixelFraction(pixelFraction_)
            {
            }

            // INTERFACE: returns the number of images in the dataset
            size_t GetImageCount() const
            {
                return original.GetImageCount();
            }

            // INTERFACE: returns the idx'th ground truth image
            ImageRefC<UnaryGroundLabel> GetGroundTruthImage(size_t idx) const
            {
                return response[idx];
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

            // Returns a vector containing references to the ground truth images
            VecCRef<ImageRefC<UnaryGroundLabel>> GetOriginalGroundVector() const
            {
                VecRef<ImageRefC<UnaryGroundLabel>> ret;

                for(size_t i = 0; i < GetImageCount(); ++i)
                    ret.push_back(original.GetGroundTruthImage(i));

                return ret;
            }

            size_t
            NumValues() const
            {
                size_t ret = 0;

                for(size_t i = 0; i < GetImageCount(); ++i)
                {
                    const auto ground = GetGroundTruthImage(i);
                    ret += (ground.Width() * ground.Height());
                }

                return ret * UnaryGroundLabel::Size;
            }

            void InitializeForCascadeLevel(size_t level, const VecCRef<ImageRefC<UnaryGroundLabel>>& prediction) const
            {
                original.InitializeForCascadeLevel(level, prediction);
            }
        };

        // Return an empty pseudo response for each image in the given dataset
        template<typename TDataSampler>
        VecCRef< ImageRef<typename TDataSampler::UnaryGroundLabel> >
        ZeroPseudoResponse(const TDataSampler& sampler)
        {
            VecRef< ImageRef<typename TDataSampler::UnaryGroundLabel> > ret;

            for(size_t i = 0; i < sampler.GetImageCount(); ++i)
            {
                auto ground = sampler.GetGroundTruthImage(i);
                ret.push_back(ImageRef<typename TDataSampler::UnaryGroundLabel>(ground.Width(), ground.Height()));
            }

            return ret;
        }

        // Returns an empty pseudo response, with image dimensions equivalent to the given data
        template<typename TLabel>
        VecCRef< ImageRef<TLabel> >
        ZeroPseudoResponse(const VecCRef< ImageRefC<TLabel> >& response)
        {
            VecRef< ImageRef<TLabel> > vec;
            std::for_each(response.begin(), response.end(), [&](const ImageRefC<TLabel>& image)
            {
                vec.push_back(ImageRef<TLabel>(image.Width(), image.Height()));
            });
            return vec;
        }

        // Returns an empty pseudo response of dimensionality equal to the provided input image
        template<typename TDataSampler>
        ImageRef<typename TDataSampler::UnaryGroundLabel>
        ZeroPseudoResponseImage(const ImageRefC<typename TDataSampler::InputLabel>& input)
        {
            return ImageRef<typename TDataSampler::UnaryGroundLabel>(input.Width(), input.Height());
        }

        // Adds a scaled pseudo response to the current pseudo response.
        template<typename TLabel>
        void AddPseudoResponse(const VecCRef< ImageRef<TLabel> >& soFar,
                               typename TLabel::ValueType eta,
                               const VecCRef< ImageRefC<TLabel> >& add)
        {
            const auto cc = TLabel::Size;

            for(size_t i = 0; i < soFar.size(); ++i)
            {
                const auto iSoFar = soFar[i];
                const auto iAdd   = add[i];
                const auto cx     = iSoFar.Width(), cy = iSoFar.Height();
                //#pragma omp parallel for

                for(int y = 0; y < cy; ++y)
                    for(int x = 0; x < cx; ++x)
                        for(int c = 0; c < cc; ++c)
                            iSoFar(x, y)[c] += eta * iAdd(x, y)[c];
            }
        }

        template<typename TLabel>
        VecCRef< ImageRef<TLabel> >
        ResponsePlusScaled(const VecCRef<ImageRefC<TLabel>>& predSoFar,
                           typename TLabel::ValueType eta,
                           const VecCRef<ImageRefC<TLabel>>& predNew,
                           const VecCRef<ImageRef<TLabel>>& sum)
        {
            const auto cc = TLabel::Size;

            for(size_t i = 0; i < predSoFar.size(); ++i)
            {
                const auto iSoFar = predSoFar[i];
                const auto iAdd   = predNew[i];
                auto       iSum   = sum[i];
                const auto cx     = iSoFar.Width(), cy = iSoFar.Height();
                //#pragma omp parallel for

                for(int y = 0; y < cy; ++y)
                    for(int x = 0; x < cx; ++x)
                        for(int c = 0; c < cc; ++c)
                            iSum(x, y)[c] = iSoFar(x, y)[c] + eta * iAdd(x, y)[c];
            }

            return sum;
        }

        template<typename TLabel>
        VecCRef< ImageRefC<TLabel> >
        ResponsePlusScaled(const VecCRef<ImageRefC<TLabel>>& predSoFar,
                           typename TLabel::ValueType eta,
                           const VecCRef<ImageRefC<TLabel>>& predNew)
        {
            VecRef<ImageRefC<TLabel>> ret;
            const auto cc = TLabel::Size;

            for(size_t i = 0; i < predSoFar.size(); ++i)
            {
                const auto iSoFar = predSoFar[i];
                const auto iAdd   = predNew[i];
                const auto cx     = iSoFar.Width(), cy = iSoFar.Height();
                auto       iSum   = ImageRef<TLabel>(cx, cy);
                //#pragma omp parallel for

                for(int y = 0; y < cy; ++y)
                    for(int x = 0; x < cx; ++x)
                        for(int c = 0; c < cc; ++c)
                            iSum(x, y)[c] = iSoFar(x, y)[c] + eta * iAdd(x, y)[c];

                ret.push_back(iSum);
            }

            return ret;
        }

        // Normalizes the pseudo response such that the largest single component has a value
        // of 1.0. This enables better comparability of the stopping criteria (gradient norm, etc.)
        // for RTF training w.r.t. pseudo responses.
        template<typename TLabel>
        void NormalizePseudoResponse(const VecCRef< ImageRef<TLabel> >& response)
        {
            const auto cc = TLabel::Size;
            typename TLabel::ValueType maxval = 0;

            for(size_t i = 0; i < response.size(); ++i)
            {
                const auto iResponse = response[i];
                const auto cx        = iResponse.Width(), cy = iResponse.Height();

                for(int y = 0; y < cy; ++y)
                    for(int x = 0; x < cx; ++x)
                        for(int c = 0; c < cc; ++c)
                            if(std::fabs(iResponse(x, y)[c]) > maxval)
                                maxval = std::fabs(iResponse(x, y)[c]);
            }

            const auto scale = 1.0 / maxval;

            for(size_t i = 0; i < response.size(); ++i)
            {
                const auto iResponse = response[i];
                const auto cx        = iResponse.Width(), cy = iResponse.Height();

                for(int y = 0; y < cy; ++y)
                    for(int x = 0; x < cx; ++x)
                        for(int c = 0; c < cc; ++c)
                            iResponse(x, y)[c] *= scale;
            }
        }

        // Forms a linear combination of the provided predictions using the provided multipliers, using sum as temporary storage
        template<typename TLabel>
        VecCRef< ImageRef<TLabel> >
        LinearCombination(const std::vector<VecCRef<ImageRefC<TLabel>>>& predictions,
                          const Eigen::Matrix<typename TLabel::ValueType, Eigen::Dynamic, 1>& multipliers,
                          const VecCRef<ImageRef<TLabel>>& sum)
        {
            assert(predictions.size() > 0);
            VecRef<ImageRefC<TLabel>> combination;
            const auto cc = TLabel::Size;

            for(size_t i = 0; i < predictions[0].size(); ++i)
            {
                const auto iSum = sum[i];
                iSum.Clear();
                const auto cx   = iSum.Width(), cy = iSum.Height();

                for(int p = 0; p < predictions.size(); ++p)
                {
                    const auto piAdd = predictions[p][i];
                    const auto scale = multipliers[p];
                    //#pragma omp parallel for

                    for(int y = 0; y < cy; ++y)
                        for(int x = 0; x < cx; ++x)
                            for(int c = 0; c < cc; ++c)
                                iSum(x, y)[c] += scale * piAdd(x, y)[c];
                }
            }

            return sum;
        }

        // Adds the scaled response for a single image
        template<typename TLabel>
        void AddPseudoResponse(const ImageRef<TLabel>& soFar,
                               typename TLabel::ValueType eta,
                               const ImageRefC<TLabel>& add)
        {
            const auto cc = TLabel::Size;
            const auto cx = soFar.Width(), cy = soFar.Height();
            //#pragma omp parallel for

            for(int y = 0; y < cy; ++y)
                for(int x = 0; x < cx; ++x)
                    for(int c = 0; c < cc; ++c)
                        soFar(x, y)[c] += eta * add(x, y)[c];
        }

        template < typename TFeatureSampler,
                 typename TDataSampler,
                 typename TUSplitCritTag          = SquaredResidualsCriterion,
                 typename TPSplitCritTag          = SquaredResidualsCriterion,
                 bool UseBasis                    = false,
                 bool UseExplicitThresholdTesting = false,
                 int CachingModel                 = WEIGHTS_AND_BASIS_PRECOMPUTED,
                 typename TLinearOperatorWeights  = LinearOperator::DefaultWeights<typename TDataSampler::UnaryGroundLabel::ValueType>>
                 class BoostingBase
                 {
             public:
                     typedef Traits < TFeatureSampler,
                     Detail::PseudoResponseAdapter<TDataSampler>,
                     TUSplitCritTag,
                     TPSplitCritTag,
                     NullPrior,
                     NullPrior,
                     UseBasis,
                     UseExplicitThresholdTesting,
                     Monitor::DefaultMonitor,
                     CachingModel,
                     TLinearOperatorWeights>						    TTraits;

                     typedef typename TTraits::ValueType                     TValue;

        static void DoNothing(int) {}

protected:

        std::vector<TValue>                                     etas;

        typedef std::function <
        ImageRef <
        typename TTraits::UnaryGroundLabel > (
            const ImageRefC <
            typename TTraits::InputLabel > &) >         TBasePredictor;
        TBasePredictor                                          BasePredictor;



        // Reads the ensemble from the file at the given path
        void DeSerialize(const std::string& fname)
        {
            std::ifstream ifs(fname);
            DeSerialize(ifs);
        }

        // Reads the ensemble from the provided input stream
        void DeSerialize(std::istream& in)
        {
            size_t numEtas;
            in >> numEtas;
            etas.resize(numEtas);

            for(size_t model = 0; model < numEtas; ++model)
            {
                in >> etas[model];
                ReadModel(in);
            }
        }

        // Predict labelings of a whole dataset, using a single specified model
        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
        RegressWith(const TDataSampler& testdb, size_t maxNumIt, TValue residualTol, size_t modelIndex) const
        {
            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());

            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = RegressWith(testdb.GetInputImage(i), maxNumIt, residualTol, modelIndex);

            return predictions;
        }

        // Default way of initializing the ensemble predictions: Start out with zero image
        static TBasePredictor DefaultBasePredictor()
        {
            return Detail::ZeroPseudoResponseImage<TDataSampler>;
        }

        // Solve line-search problem, which determines the loss and the new pseudo response (anti-gradient w.r.t. prediction)
        template<typename TLossTag>
        TValue LineSearch(const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& ground,
                          const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& predSoFar,
                          const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& predNew)
        {
            typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
            auto sum = Detail::ZeroPseudoResponse(ground);
            const auto f = [&](const TVector & eta) -> TValue
            {
                return Loss::Loss<TTraits, TLossTag>::Objective(ground, Detail::ResponsePlusScaled(predSoFar, eta[0], predNew, sum));
            };
            TVector eta(1);
            eta << TValue(0);
            Minimization::CMAESMinimize<TValue, 0>(f, eta, Loss::Loss<TTraits, TLossTag>::MaxLineSearchEvaluations());
            return eta[0];
        }

        // Base implementation of iterative gradient-based boosting.
        template<typename TResidualLossTag, typename TObjectiveLossTag, size_t numModels>
        void BaseLearn(const TDataSampler& traindb,
                       const std::function<void(const Detail::PseudoResponseAdapter<TDataSampler>&)>& fitAndAddModel,
                       size_t maxNumIt=100,
                       TValue residualTol=1e-4,
                       TValue subsampleFactor = 0.5,
                       TValue boostShrinking = 1.0,
                       const std::function<void (int)>& stageCompleteCallback = BoostingBase <TFeatureSampler,
                       TDataSampler,
                       TUSplitCritTag,
                       TPSplitCritTag,
                       UseBasis,
                       UseExplicitThresholdTesting,
                       CachingModel,
                       TLinearOperatorWeights >::DoNothing)
        {
            // Ensure we're adding at least one model
            assert(numModels > 0);
            // Typedef for the loss class
            typedef Loss::Loss<TTraits, TObjectiveLossTag> TObjectiveLoss;
            typedef Loss::Loss<TTraits, TResidualLossTag> TResidualLoss;
            // Initialize pseudo response
            auto pseudoResponse  = Detail::ZeroPseudoResponse<TDataSampler>(traindb);
            auto adapter         = Detail::PseudoResponseAdapter<TDataSampler>(traindb, pseudoResponse, subsampleFactor);
            auto ground          = adapter.GetOriginalGroundVector();
            auto predSoFar       = Regress(traindb, maxNumIt, residualTol);
            const auto normC     = TObjectiveLoss::NormalizationConstant(ground);
            const auto initLoss  = TObjectiveLoss::Objective(ground, predSoFar) / normC;
            TResidualLoss::PseudoResponse(ground, predSoFar, pseudoResponse);

            if(EnsembleSize() > 0)
            {
                TTraits::Monitor::Report("Boosting: Starting training of %u additional models.\n", numModels);
                TTraits::Monitor::Report("Boosting: Current %s loss of %d-ensemble is %.6f.\n", TObjectiveLoss::Name(), EnsembleSize(), initLoss);
            }
            else
            {
                TTraits::Monitor::Report("Boosting: Starting training of %u models.\n", numModels);
                TTraits::Monitor::Report("Boosting: Initial %s loss is %.6f.\n", TObjectiveLoss::Name(), initLoss);
            }

            // Start where were previously left
            size_t model          = EnsembleSize();
            const auto lastModel  = model + numModels - 1;
            TValue loss           = TValue(0);

            do
            {
                TTraits::Monitor::Report("Boosting: Now training model no. %u.\n", model);
                // Choose the variables used for training
                adapter.ResampleVariables();
                adapter.InitializeForCascadeLevel(model, predSoFar);
                // Divide the pseudo response by the largest absolute value occurring in it to improve numerical stability for some losses
                //Detail::NormalizePseudoResponse(pseudoResponse);
                // Add new model and fit it to subset of current pseudo response
                fitAndAddModel(adapter);
                // Predict using the current model
                TTraits::Monitor::Report("Boosting: Predicting using model no. %u.\n", model);
                auto predNew = RegressWith(traindb, maxNumIt, residualTol, model);
                // Solve line-search problem, which determines the new loss
                TTraits::Monitor::Report("Boosting: Performing line search.\n");
                auto eta = LineSearch<TObjectiveLossTag>(ground, predSoFar, predNew);
                // Update ensemble and its current prediction
                predSoFar = Detail::ResponsePlusScaled(predSoFar, boostShrinking * eta, predNew);
                etas.push_back(boostShrinking * eta);
                // Evaluate ensemble and compute pseudo-response
                loss = TObjectiveLoss::Objective(ground, predSoFar) / normC;
                TResidualLoss::PseudoResponse(ground, predSoFar, pseudoResponse);
                TTraits::Monitor::Report("Boosting: Loss of shrunk ensemble (eta = %.4f) is %.6f.\n", (boostShrinking * eta), loss);

                stageCompleteCallback(model);
            }
            while(++model <= lastModel);

            TTraits::Monitor::Report("Boosting: Finished training of %d-ensemble.\n", EnsembleSize());
            TTraits::Monitor::Report("Boosting: Observe that %s loss decreased from %.6f to %.6f - yay!\n", TObjectiveLoss::Name(), initLoss, loss);
        }


        /* Pure virtual functions that must be implemented by any concrete subclass */

        // Read the next model from the provided stream and add it to the ensemble
        virtual void ReadModel(std::istream& in) = 0;

        // Write the indicated model to the provided stream
        virtual void WriteModel(std::ostream& out, size_t model) const = 0;

        // Predict the labelling of a single image using the specified model
        virtual ImageRefC<typename TTraits::UnaryGroundLabel>
        RegressWith(const ImageRefC<typename TTraits::InputLabel>& input, size_t maxNumIt, TValue residualTol, size_t modelIndex) const = 0;

public:

        // Default constructor
        BoostingBase(TBasePredictor BasePredictor_)
            : BasePredictor(BasePredictor_)
        {
        }

        // Returns the number of models in the ensemble (does not include the base predictor, if any)
        size_t EnsembleSize() const
        {
            return etas.size();
        }

        // Compute the ensemble prediction for a single input image, using the first numModels models (default is -1: use all models).
        // If numModels is set to 0, only the "base predictor" is used.
        ImageRefC<typename TTraits::UnaryGroundLabel>
        Regress(int idx, const TDataSampler& testdb, size_t maxNumIt=100, TValue residualTol=1e-4, int numModels = -1) const
        {
            auto prediction = BasePredictor(testdb.GetInputImage(idx));
            size_t numPreds = numModels < 0 ? EnsembleSize() : numModels;

            for(size_t model = 0; model < numPreds; ++model)
            {
                testdb.InitializeForCascadeLevel(idx, model, prediction);
                auto response = RegressWith(testdb.GetInputImage(idx), maxNumIt, residualTol, model);
                Detail::AddPseudoResponse(prediction, etas[model], response);
            }

            return prediction;
        }

        // Compute the ensemble prediction for a whole dataset, using the first numModels models (default is -1: use all models)
        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
        Regress(const TDataSampler& testdb, size_t maxNumIt=100, TValue residualTol=1e-4, int numModels = -1) const
        {
            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());

            for(int i = 0; i < predictions.size(); ++i)
            {
                predictions[i] = Regress(i, testdb, maxNumIt, residualTol, numModels);
            }

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

        // Store the ensemble in a file at the given path
        void Serialize(const std::string& fname) const
        {
            std::ofstream ofs(fname);
            Serialize(ofs);
        }

        // Write the ensemble to the given output stream
        void Serialize(std::ostream& out) const
        {
            out << EnsembleSize() << std::endl;

            for(size_t model = 0; model < EnsembleSize(); ++model)
            {
                out << etas[model] << std::endl;
                WriteModel(out, model);
            }
        }
                 };

    } // namespace Detail

    // Class BoostedRTF
    // ================
    //
    // The main interface to the RTF boosting functionality. The following main use cases are supported:
    //
    // 1) Instantiate a new ensemble and train it to fit a given training set
    //
    //    a) Create an empty instance of BoostedRTF:
    //
    //         Boosting::BoostedRTF<MyFeatureSampler, MyDataset> rtf;
    //
    //    b) Add the factor types, for instance for a 4-connected model with a single unary type:
    //
    //         VecRef<Vector2D<int>> hOffsets;
    //         hOffsets.push_back(Vector2D<int>(0, 0));
    //         hOffsets.push_back(Vector2D<int>(1, 0));
    //
    //         VecRef<Vector2D<int>> vOffsets;
    //         vOffsets.push_back(Vector2D<int>(0, 0));
    //         vOffsets.push_back(Vector2D<int>(0, 1));
    //
    //         rtf.AddUnaryFactorType(100, 7, 16, 1e-2, 1e2);
    //         rtf.AddPairwiseFactorType(hOffsets, 100, 4, 32, 1e-2, 1e2);
    //         rtf.AddPairwiseFactorType(vOffsets, 100, 4, 32, 1e-2, 1e2);
    //
    //    c) Learn an ensemble of 10 regression tree fields that optimizes MSE loss
    //
    //         rtf.Learn<Loss::MSE, 10>(ds, 30, 50, 1e-6, 0.5, 0.3);
    //
    // 2) Save an existing ensemble of RTFs that was previously trained
    //
    //         rtf.Serialize("model.dump");
    //
    // 3) Load an ensemble of RTFs that was previously trained and subsequently serialized.
    //
    //         Boosting::BoostedRTF<MyFeatureSampler, MyDataset> loaded("model.dump");
    //
    // Please see the individual method declarations for full details about the parameters.
    //
    // Using a "base predictor":
    // -------------------------
    // Often, it is convenient to start out from an existing blackbox method that can produce
    // a ground truth labeling from a given input image reasonably well, and then add a number
    // of RTFs so as to remove the remaining noise in a loss-specific manner.
    // Towards this end, each of the constructors supports passing in such a "base predictor",
    // which must be a function that takes an input image as its argument and returns a
    // ground truth image according to the model traits. Unless specified otherwise, a default
    // "base predictor" will be chosen that simply returns a *zero* ground truth, i.e. has no
    // effect at all.
    // Note: The "base predictor" will not be serialized when dumping the model to a file,
    // so you need to specify it again when "loading" the model by passing the corresponding
    // function as an argument to the deserialization constructor.
    //
    template < typename TFeatureSampler,
             typename TDataSampler,
             typename TSplitCritTag           = SquaredResidualsCriterion,
             bool UseBasis                    = false,
             bool UseExplicitThresholdTesting = false,
             int CachingModel                 = WEIGHTS_AND_BASIS_PRECOMPUTED,
             typename TLinearOperatorWeights  = LinearOperator::DefaultWeights<typename TDataSampler::UnaryGroundLabel::ValueType>>
         class RTF : public Detail::BoostingBase<TFeatureSampler, TDataSampler, TSplitCritTag, TSplitCritTag, UseBasis, UseExplicitThresholdTesting, CachingModel, TLinearOperatorWeights>
             {
             public:
                 typedef Detail::BoostingBase<TFeatureSampler, TDataSampler, TSplitCritTag, TSplitCritTag, UseBasis, UseExplicitThresholdTesting, CachingModel, TLinearOperatorWeights> Base;
                 typedef typename Base::TTraits TTraits;
                 typedef typename Base::TValue TValue;
                 typedef typename Base::TBasePredictor TBasePredictor;

             private:
                 typedef std::function<typename TTraits::UnaryFactorType(int)>    UnaryTypeInstantiator;
                 typedef std::function<typename TTraits::PairwiseFactorType(int)> PairwiseTypeInstantiator;
                 typedef std::function<typename TTraits::LinearOperatorRef(int)>  LinearOperatorInstantiator;

                 std::vector<typename TTraits::UnaryFactorTypeVector>			 utypes;
                 std::vector<typename TTraits::PairwiseFactorTypeVector>			 ptypes;
                 std::vector<typename TTraits::LinearOperatorVector>				 ltypes;

                 std::vector<Learning::Detail::FactorTypeInfo<TValue>>			 uinfos;
                 std::vector<UnaryTypeInstantiator>								 ucreat;
                 std::vector<Learning::Detail::FactorTypeInfo<TValue>>			 pinfos;
                 std::vector<PairwiseTypeInstantiator>							 pcreat;
                 std::vector<LinearOperatorInstantiator>							 lcreat;

                 void AddFactorTypes()
    {
        typename TTraits::UnaryFactorTypeVector ut;

        for(size_t u = 0; u < ucreat.size(); ++u)
            ut.push_back(ucreat[u](utypes.size()));

        utypes.push_back(ut);
        typename TTraits::PairwiseFactorTypeVector pt;

        for(size_t p = 0; p < pcreat.size(); ++p)
            pt.push_back(pcreat[p](ptypes.size()));

        ptypes.push_back(pt);

        typename TTraits::LinearOperatorVector lt;

        for( size_t l = 0; l < lcreat.size(); ++l )
            lt.push_back(lcreat[l](ltypes.size()));

        ltypes.push_back(lt);
    }

protected:

    // Read the next model from the provided stream and add it to the ensemble
    void ReadModel(std::istream& in)
    {
        utypes.resize(utypes.size() + 1);
        ptypes.resize(utypes.size() + 1);
        ltypes.resize(ltypes.size() + 1);
        Serialization::ReadModel<TTraits>(in, utypes.back(), ptypes.back(), ltypes.back());
    }

    // Write the indicated model to the provided stream
    void WriteModel(std::ostream& out, size_t model) const
    {
        assert(utypes.size() > model && ptypes.size() > model);
        Serialization::WriteModel<TTraits>(out, utypes[model], ptypes[model], ltypes[model]);
    }

    // Predict the labelling of a single image using the specified model
    ImageRefC<typename TTraits::UnaryGroundLabel>
    RegressWith(const ImageRefC<typename TTraits::InputLabel>& input, size_t maxNumIt, TValue residualTol, size_t model) const
    {
        return Classify::Predict<TTraits>(utypes[model], ptypes[model], ltypes[model], input, residualTol, maxNumIt);
    }

public:

    // Deserialization constructor: Load RTF from the file at the given path
    RTF(const std::string& fname,
        TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
        Base::DeSerialize(fname);
    }

    // Deserialization constructor: Load RTF from the file at the given path
    RTF(const char* fname,
        TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
        Base::DeSerialize(fname);
    }

    // Deserialization constructor: Load RTF from provided stream
    RTF(std::istream& in,
        TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
        Base::DeSerialize(in);
    }

    // Default constructor
    RTF(TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
    }

    // Adds a unary factor type. The parameters specify the characteristics of the underlying regression
    // tree that is to be trained, as well as regularization of the model parameters.
    void AddUnaryFactorType(int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                            TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                            TValue linearRegularizationC = 0, TValue quadraticRegularizationC = 0, TValue purityEpsilon = 0)
    {
        ucreat.push_back([ = ](int)
        {
            return Learning::MakeUnaryFactorType<TTraits>(smallestEigenValue, largestEigenValue,
                    linearRegularizationC, quadraticRegularizationC);
        });
        uinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                         nMinDataPointsForSplitConsideration, purityEpsilon));
    }

    // Similar to the above, but for pairwise factor types. The offsets vector specifies which variables (relative to
    // a given pixel) are covered by the factor.
    void AddPairwiseFactorType(const Vector2D<int>& offsets,
                               int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                               TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                               TValue linearRegularizationC = 0, TValue quadraticRegularizationC = 0, TValue purityEpsilon = 0)
    {
        VecRef<Vector2D<int>> offvec;
        offvec.push_back(Vector2D<int>(0, 0));   // the first variable is always 0,0 by convention
        offvec.push_back(offsets);               // offsets of the second variable, relative to 0,0
        pcreat.push_back([ = ](int)
        {
            return Learning::MakePairwiseFactorType<TTraits>(offvec,
                    smallestEigenValue, largestEigenValue, -1,
                    linearRegularizationC, quadraticRegularizationC);
        });
        pinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                         nMinDataPointsForSplitConsideration, purityEpsilon));
    }

    // Adds a custom linear operator. The user must provide a function object returning a new
    // instance of the linear operator (LinearOperatorRef) upon invocation.
    void AddLinearOperator(LinearOperatorInstantiator instantiator)
    {
        lcreat.push_back(instantiator);
    }

    // Learns an ensemble of 'numRTFs' regression tree fields such as to optimize the loss
    // specified via the tag 'TLossTag'. At each step, a line-search is performed to find
    // the optimal multiplier for the newly added model.
    // Multiple invocations of the method are supported. If the ensemble is non-empty upon
    // invocation, the initial pseudo-response is determined by first regressing using the
    // existing ensemble and the previously determined multipliers associated with the models.
    // The parameters starting with 'rtf' are passed on to the joint training routine that
    // trains the individual RTFs and specify how, and to what degree of optimality,
    // the model parameters of the individual RTFs should be optimized.
    // The 'boostShrinking' parameter scales the optimal step size for each added RTF and should
    // lie between 0.0 and 1.0 (full step). Empirically, it has been observed that shrinking leads
    // to an ensemble that generalizes better, but it takes a larger number of RTFs to achieve
    // low loss on the training data and may thus be impractical for many applications. You can
    // think of shrinking as a way of protecting against overfitting - use with care and only if
    // you observe that your ensemble overfits. An alternative measure to protect against overfitting
    // is simply to reduce the number of classifiers in the ensemble. Finally, the individual RTFs
    // could be chosen less powerful (by restricting the tree depths, for instance).
    // The most successful strategy depends heavily on the problem at hand.
    template<typename TLossTag, size_t numRTFs>
    void Learn(const TDataSampler& traindb,
               size_t rtfMaxNumOptimItPerRound = 50,
               size_t rtfMaxNumOptimItFinal    = 50,
               TValue rtfFinalBreakEps         = 1e-2,
               TValue rtfSubsampleFactor       = 0.3,
               TValue boostShrinking           = 1.0)
    {
        // Ensure that at least one factor type has been added prior to invocation of this method
        assert(uinfos.size() >= 1 || pinfos.size() >= 1 || lcreat.size() >= 1);
        return this->template BaseLearn<TLossTag, TLossTag, numRTFs>(traindb, [&](const Detail::PseudoResponseAdapter<TDataSampler>& adapter)
        {
            // Add new RTF and fit it to subset of current pseudo response
            this->AddFactorTypes();
            Learning::LearnTreesAndWeightsJointly<TTraits, true, 5>(utypes.back(), uinfos, ptypes.back(), pinfos, adapter,
                    rtfMaxNumOptimItPerRound, rtfMaxNumOptimItFinal, rtfFinalBreakEps);
        }, rtfSubsampleFactor, boostShrinking);
    }

    template<typename TBoostLossTag, typename TDiscriminativeLossTag, size_t numRTFs>
    void LearnDiscriminative(const TDataSampler& traindb,
                             size_t rtfMaxNumOptimItPerRound = 50,
                             size_t rtfMaxNumOptimItFinal    = 50,
                             TValue rtfFinalBreakEps         = 1e-2,
                             TValue rtfSubsampleFactor       = 0.3,
                             bool   stagedTraining           = false,
                             size_t cgMaxNumIt               = 100,
                             TValue cgBreakEps               = 1e-3,
                             TValue boostShrinking           = 1.0,
                             const std::function<void (int)>& stageCompleteCallback = Base::DoNothing)
    {
        // Ensure that at least one factor type has been added prior to invocation of this method
        assert(uinfos.size() >= 1 || pinfos.size() >= 1 || lcreat.size() >= 1);
        return this->template BaseLearn<TBoostLossTag, TDiscriminativeLossTag, numRTFs>(traindb, [&](const Detail::PseudoResponseAdapter<TDataSampler>& adapter)
        {
            // Add new RTF and fit it to subset of current pseudo response
            this->AddFactorTypes();
            Learning::LearnTreesAndWeightsJointlyDiscriminative<TTraits, TDiscriminativeLossTag, true, 32>(
                utypes.back(), uinfos, ptypes.back(), pinfos, ltypes.back(),
                adapter,
                rtfMaxNumOptimItPerRound, rtfMaxNumOptimItFinal, rtfFinalBreakEps,
                stagedTraining,
                cgMaxNumIt, cgBreakEps);
        }, cgMaxNumIt, cgBreakEps, rtfSubsampleFactor, boostShrinking, stageCompleteCallback);
    }

             };

    // Class BoostedForest
    // ================
    //
    // The main interface to the tree boosting functionality. The following main use cases are supported:
    //
    // 1) Instantiate a new ensemble and train it to fit a given training set
    //
    //    a) Create an empty instance of BoostedForest:
    //
    //         Boosting::BoostedForest<MyFeatureSampler, MyDataset> forest;
    //
    //    c) Learn an ensemble of 10 regression trees that optimizes MSE loss
    //
    //         forest.Learn<Loss::MSE, 10>(ds, 200, 16);
    //
    // 2) Save an existing ensemble of trees that was previously trained
    //
    //         forest.Serialize("model.dump");
    //
    // 3) Load an ensemble of trees that was previously trained and subsequently serialized.
    //
    //         Boosting::BoostedForest<MyFeatureSampler, MyDataset> loaded("model.dump");
    //
    // Please see the individual method declarations for full details about the parameters.
    //
    // Using a "base predictor":
    // -------------------------
    // Often, it is convenient to start out from an existing blackbox method that can produce
    // a ground truth labeling from a given input image reasonably well, and then add a number
    // of trees so as to remove the remaining noise in a loss-specific manner.
    // Towards this end, each of the constructors supports passing in such a "base predictor",
    // which must be a function that takes an input image as its argument and returns a
    // ground truth image according to the model traits. Unless specified otherwise, a default
    // "base predictor" will be chosen that simply returns a *zero* ground truth, i.e. has no
    // effect at all.
    // Note: The "base predictor" will not be serialized when dumping the model to a file,
    // so you need to specify it again when "loading" the model by passing the corresponding
    // function as an argument to the deserialization constructor.
    //
    template < typename TFeatureSampler,
             typename TDataSampler,
             typename TSplitCritTag           = SquaredResidualsCriterion,
             bool UseExplicitThresholdTesting = false >
         class BoostedForest : public Detail::BoostingBase<TFeatureSampler, TDataSampler, TSplitCritTag, TSplitCritTag, false, UseExplicitThresholdTesting>
             {
             private:
                 typedef Detail::BoostingBase<TFeatureSampler, TDataSampler, TSplitCritTag, TSplitCritTag, false, UseExplicitThresholdTesting> Base;
                 typedef typename Base::TTraits TTraits;
                 typedef typename Base::TValue TValue;
                 typedef typename Base::TBasePredictor TBasePredictor;

                 typedef std::function<typename TTraits::UnaryFactorType()>    UnaryTypeInstantiator;
                 typedef std::function<typename TTraits::PairwiseFactorType()> PairwiseTypeInstantiator;

                 typename TTraits::UnaryTreeRefVector                          trees;

             protected:

                 // Read the next model from the provided stream and add it to the ensemble
                 void ReadModel(std::istream& in)
    {
        typename TTraits::UnaryTreeRef ref;
        Serialization::ReadTree(in, ref);
        trees.push_back(ref);
    }

    // Write the indicated model to the provided stream
    void WriteModel(std::ostream& out, size_t tree) const
    {
        assert(trees.size() > tree);
        Serialization::WriteTree(out, trees[tree]);
    }

    // Predict the labelling of a single image using the specified model
    ImageRefC<typename TTraits::UnaryGroundLabel>
    RegressWith(const ImageRefC<typename TTraits::InputLabel>& input, size_t maxNumIt, TValue residualTol, size_t model) const
    {
        assert(trees.size() > model);
        return Classify::Regress<TTraits>(trees[model], input);
    }

public:

    // Deserialization constructor: Load forest from the file at the given path
    BoostedForest(const std::string& fname,
                  TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
        Base::DeSerialize(fname);
    }

    // Deserialization constructor: Load forest from the file at the given path
    BoostedForest(const char* fname,
                  TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
        Base::DeSerialize(fname);
    }

    // Deserialization constructor: Load forest from provided stream
    BoostedForest(std::istream& in,
                  TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
        Base::DeSerialize(in);
    }

    // Default constructor
    BoostedForest(TBasePredictor BasePredictor_ = Base::DefaultBasePredictor())
        : Base::BoostingBase(BasePredictor_)
    {
    }

    // Learns an ensemble of 'numTrees' regression trees such as to optimize the loss
    // specified via the tag 'TLossTag'. At each step, a line-search is performed to find
    // the optimal multiplier for the newly added model.
    // Multiple invocations of the method are supported. If the ensemble is non-empty upon
    // invocation, the initial pseudo-response is determined by first regressing using the
    // existing ensemble and the previously determined multipliers associated with the models.
    // The parameters starting with 'tree' are passed on to the tree training routine.
    // The 'boostShrinking' parameter scales the optimal step size for each added tree and should
    // lie between 0.0 and 1.0 (full step). Empirically, it has been observed that shrinking leads
    // to an ensemble that generalizes better, but it takes a larger number of trees to achieve
    // low loss on the training data and may thus be impractical for many applications. You can
    // think of shrinking as a way of protecting against overfitting - use with care and only if
    // you observe that your ensemble overfits. An alternative measure to protect against overfitting
    // is simply to reduce the number of classifiers in the ensemble. Finally, the individual trees
    // could be chosen less powerful (by restricting the tree depths, for instance).
    // The most successful strategy depends heavily on the problem at hand.
    template<typename TLossTag, size_t numTrees>
    void Learn(const TDataSampler& traindb,
               int    treeFeatureCount,
               int    treeDepthLevels,
               int    treeMinDataPointsForSplitConsideration = 32,
               TValue treePurityEpsilon                      = 0.0,
               TValue treeSubsampleFactor                    = 0.3,
               TValue boostShrinking                         = 1.0,
               const std::function<void (int)>& stageCompleteCallback = Base::DoNothing)
    {
        this->template BaseLearn<TLossTag, TLossTag, numTrees>(traindb, [&](const Detail::PseudoResponseAdapter<TDataSampler>& adapter)
        {
            // Add new tree and fit it to subset of current pseudo response
            trees.push_back(Training::LearnUnaryRegressionTreeSubsample<TTraits>(adapter, treeFeatureCount, treeDepthLevels,
                            treeMinDataPointsForSplitConsideration, treePurityEpsilon));
        }, 0, 0, treeSubsampleFactor, boostShrinking, stageCompleteCallback);
    }
             };

} // namespace Boosting

#endif // _H_BOOSTING_H_
