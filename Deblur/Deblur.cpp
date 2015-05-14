namespace Deblurring
{
  template<typename TFeature, typename TUnaryGroundLabel> class BlurKernelOperator;
}
#define TYPE_BLUR_KERNEL_OPERATOR -1
#define INSTANTIATE_CUSTOM_OPERATOR(typeid) new Deblurring::BlurKernelOperator<TFeature, TUnaryGroundLabel>()

#define PARALLEL_SYSMATRIX_ALLOCATION

#include "RTF/Loss.h"
#include "RTF/Stacked.h"
#include "RTF/LinearOperator.h"
#include "RTF/Monitor.h"
#include "Dataset.h"
#include "Feature.h"
#include "Operator.h"
#include <itkFFTWGlobalConfiguration.h>

typedef Deblurring::Dataset Dataset_t;
typedef Deblurring::FeatureSampler FeatureSampler_t;
typedef Deblurring::FeatureSampler::TFeature Feature_t;
typedef Deblurring::Dataset::UnaryGroundLabel Label_t;
typedef LinearOperator::DefaultWeights<Label_t::ValueType> DefaultWeights_t;
typedef Stacked::RTF<FeatureSampler_t,
		       Dataset_t,
		       GradientNormCriterion, // split based on gradient norm
		       true, // use linear basis
		       true, // use explicit threshold checking in tree splits
		       WEIGHTS_AND_BASIS_PRECOMPUTED> RTF_t;
typedef Monitor::DefaultMonitor MyMonitor; // console output channel

// set path to base directory
boost::filesystem::path path = "";

int main ()
{
    // set deterministic FFT
    itk::FFTWGlobalConfiguration::SetPlanRigor("FFTW_ESTIMATE");

    // // settings for standard RTF cascade
    // RTF_t rtf((path / "models" / "standard.txt").string());
    // Dataset_t testds((path / "demo").string(), "test");
    // size_t CascadeDepth = 2; // standard RTF

    // settings for interleaved RTF cascade
    RTF_t rtf((path / "models" / "interleaved.txt").string());
    Dataset_t testds((path / "demo").string(), "test");
    size_t CascadeDepth = 3;
    testds.Interleaved();

    // regress on the test data
    int maxNumItCG = 10000;
    double residualTolCG = 1e-6;
    size_t endLevel = CascadeDepth-1;
    auto prediction = rtf.Regress(testds, maxNumItCG, residualTolCG, endLevel);

    // write out the predictions
    boost::filesystem::path outp;
    outp = path / "demo" / "predictions" / std::to_string(CascadeDepth);

    MyMonitor::Report("Writing predictions ...\n");
    for(int i = 0; i < testds.GetImageCount(); ++i)
    {
	int offset = (testds.GetKernelSize(i)-1)/2;
	testds.SaveGroundTruthImage(prediction[i], (outp / (testds.GetImageName(i) +
							    "_deblur" + ".png")).string(), offset);
	testds.SaveGroundTruthImage(prediction[i], (outp / (testds.GetImageName(i) +
							    "_deblur" + ".dlm")).string(), offset);
    }
}
