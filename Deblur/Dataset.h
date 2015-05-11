#ifndef _H_DATASET_H_
#define _H_DATASET_H_

#include <boost/filesystem.hpp>

#include "RTF/Training.h"
#include "Filters.h"

#include "QiDAGM12Filterbank.h"
#define Filterbank QiDAGM12Filterbank


namespace Deblurring
{
    class IOException : public std::exception
    {
    private:
	std::string w;
    public:
    IOException(const std::string& what_) : w(what_) {}

	virtual const char *what() const throw()
	{
	    return w.c_str();
	}

	virtual ~IOException() throw() {}
    };

    class Dataset

    {
    public:
	typedef double FPType;
	typedef Training::LabelVector<FPType, 1>   UnaryGroundLabel;
	typedef Training::LabelVector<FPType, 2>   PairwiseGroundLabel;
	typedef Training::LabelVector<FPType,
	    1 + // input image
	    1 + // previous output of the model (deblurred image)
	    Filterbank::filter_count + // filter responses, arranged as RGB triplets
	    1   // current estimate of the blur kernel
	    > InputLabel;

	enum InputIndex {IDX_INPUT_IMAGE = 0,
			  IDX_SHARP_IMAGE = 1,
			  IDX_FILTER_RESPONSES = IDX_SHARP_IMAGE + 1,
			  IDX_BLUR_KERNEL = IDX_FILTER_RESPONSES + Filterbank::filter_count};

    Dataset(const std::string& path_, const std::string& type_)
	: path(path_), type(type_), interleaved(false)
	{
	    ReadDescriptor();
	}

	// INTERFACE: returns the number of images in the dataset
	size_t GetImageCount() const
	{
	    return inputImages.size();
	}

	// INTERFACE: returns the idx'th input image.
	ImageRefC<InputLabel> GetInputImage(size_t idx) const
	{
	    assert(idx < inputImages.size());
	    if( ! inputImages[idx] )
	    {
		inputImages[idx] = LoadInputImageAndKernel(idx);
	    }
	    return inputImages[idx];
	}

	// INTERFACE: returns the idx'th kernel size
	int  GetKernelSize(size_t idx) const
	{
	    assert(idx < kernelSizes.size());
	    return kernelSizes[idx];
	}

	int GetKernelSize(const ImageRef<InputLabel> input) const
	{
	    return (int) input(input.Width()-1, input.Height()-1)[IDX_BLUR_KERNEL];
	}

	// INTERFACE: the method is called prior to each training stage
	void InitializeForCascadeLevel(size_t level, VecCRef<ImageRefC<UnaryGroundLabel>> previousPrediction) const
	{
	    for( size_t idx = 0; idx < GetImageCount(); ++idx )
	    {
		// Read image into memory
		GetInputImage(idx);

		if( level == 0 )
		    PreProcessInput(inputImages[idx], fileNames[idx]);
		else
		{
		    PreProcessInput(inputImages[idx], fileNames[idx], previousPrediction[idx]);
		    SaveKernel(inputImages[idx], KernelPath(idx, level+1) + ".dlm");
		}
	    }
	}

	std::string GetImageName(size_t idx) const
	{
	    assert( idx < fileNames.size() );
	    return fileNames[idx];
	}

	static void SaveGroundTruthImage(const ImageRefC<UnaryGroundLabel>& ground,
					 const std::string& path, const int offset = 0)
	{
	    if( path.back() == 'g' || path.back() == 'G' )
		SaveGroundTruthImagePNG(ground, path, offset);
	    else
		SaveGroundTruthImageDLM(ground, path, offset);
	}

	static void SaveInputImage(const ImageRefC<InputLabel>& input, const std::string& path)
	{
	    if( path.back() == 'g' || path.back() == 'G' )
		return SaveInputImagePNG(input, path);
	    else
		return SaveInputImageDLM(input, path);
	}

	static void SaveGroundTruthImagePNG(const ImageRefC<UnaryGroundLabel>& ground,
					    const std::string& path, const int offset)
	{
	    ImageRef<unsigned char> img(ground.Width() - 2*offset, ground.Height() - 2*offset);

	    for( int y = 0; y < img.Height(); ++y )
		for( int x = 0; x < img.Width(); ++x )
		    *(img.Ptr(x,y)) = (unsigned char) (std::max(0.0f, std::min((float)
									       ground(x+offset,y+offset)[0], 1.0f))*255.0f + .5f );
	    Utility::WritePNG(img, path);
	}

	static void SaveGroundTruthImageDLM(const ImageRefC<UnaryGroundLabel>& ground,
					    const std::string& path, const int offset)
	{
	    ImageRef<float> img(ground.Width() - 2*offset, ground.Height() - 2*offset);
	    for( int y = 0; y < img.Height(); ++y )
		for( int x = 0; x < img.Width(); ++x )
		    *(img.Ptr(x,y)) = (float) ground(x+offset,y+offset)[0];
	    Utility::WriteDLM(img, path);
	}

	static void SaveInputImagePNG(const ImageRefC<InputLabel>& input, const std::string& path)
	{
	    ImageRef<unsigned char> img(input.Width(), input.Height());
	    for( int y = 0; y < img.Height(); ++y )
		for( int x = 0; x < img.Width(); ++x )
		    *(img.Ptr(x,y)) = (unsigned char) ( std::max(0.0f, std::min((float) input(x,y)[0], 1.0f))*255.0f + .5f );
	    Utility::WritePNG(img, path);
	}

	static void SaveInputImageDLM(const ImageRefC<InputLabel>& input, const std::string& path)
	{
	    ImageRef<float> img(input.Width(), input.Height());
	    for( int y = 0; y < img.Height(); ++y )
		for( int x = 0; x < img.Width(); ++x )
		    *(img.Ptr(x,y)) = input(x,y)[0];
	    Utility::WriteDLM(img, path);
	}

	std::string GetImagePath(size_t idx, const std::string& prefix) const
	{
	    assert(idx < GetImageCount());

	    boost::filesystem::path bpath = path;
	    std::string dlmPath = (bpath / prefix / (fileNames[idx] + ".dlm")).string();
	    std::string pngPath = (bpath / prefix / (fileNames[idx] + ".png")).string();

	    if( boost::filesystem::exists(dlmPath) ) {
		return dlmPath;
	    } else {
		if( ! boost::filesystem::exists(pngPath) )
		    throw IOException("Can't find " + prefix + " image: " + pngPath);
		else
		    return pngPath;
	    }
	    return pngPath;
	}

	std::string GroundTruthImagePath(size_t idx) const
	{
	    return GetImagePath(idx, "labels");
	}

	std::string InputImagePath(size_t idx) const
	{
	    return GetImagePath(idx, "images");
	}

	std::string InputKernelPath(size_t idx) const
	{
	    boost::filesystem::path bpath = path;
	    std::string dlmPath = (bpath / "initial" / (fileNames[idx] + "_kernel.dlm")).string();

	    if( boost::filesystem::exists(dlmPath) )
		return dlmPath;
	    else
		throw IOException("Can't find: " + dlmPath);
	}

	std::string KernelPath(size_t idx, size_t level) const
	{
	    boost::filesystem::path bpath = path;
	    return (bpath / "predictions" / std::to_string(level) / (fileNames[idx] + "_kernel")).string();
	}

	ImageRef<InputLabel> LoadInputImageAndKernel(const size_t idx) const
	{
	    // load kernel
	    std::string kerPath = InputKernelPath(idx);
	    ImageRefC<UnaryGroundLabel> tmpKer = LoadKernelDLM(kerPath);
	    if (tmpKer.Width() != tmpKer.Height())
	    {
		throw IOException("Unexpected asymmetric kernel size.");
	    }
	    const int kerSz = tmpKer.Width();
	    kernelSizes.push_back(kerSz);

          // load image
	    ImageRef<InputLabel> ret;
	    std::string imPath = InputImagePath(idx);
	    if( imPath.back() == 'g' || imPath.back() == 'G' )
		ret = LoadInputImagePNG(imPath);
	    else
		ret = LoadInputImageDLM(imPath);

	    ImageType::Pointer itkpad = ImageType::New();
	    GetInputAsITKImage(ret, itkpad);
	    const int offset = (kerSz - 1)/2;
	    NeumannPad(itkpad, offset);
	    ImageType::SizeType sz = GetSize(itkpad);

	    ImageRef<InputLabel> retpad(sz[0], sz[1]);
	    retpad.Clear();

	    // store image
	    ImageType::IndexType pixelIndex;
	    const int cx = retpad.Width();
	    const int cy = retpad.Height();
#pragma omp parallel for
	    for( int y = 0; y < cy; ++y )
		for( int x = 0; x < cx; ++x )
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    FPType value = itkpad->GetPixel(pixelIndex);
		    retpad(x,y)[IDX_INPUT_IMAGE] = value;
		    retpad(x,y)[IDX_SHARP_IMAGE] = value;
		}

	    // store kernel
#pragma omp parallel for
	    for( int y = 0; y < kerSz; ++y )
		for( int x = 0; x < kerSz; ++x )
		{
		    retpad(x,y)[IDX_BLUR_KERNEL] = tmpKer(x,y)[0];
		}

	    // store kernel size
	    retpad(cx-1, cy-1)[IDX_BLUR_KERNEL] = kerSz;

	    return retpad;
	}

	ImageRef<InputLabel> LoadInputImagePNG(const std::string& path) const
	{
	    std::cerr << path << std::endl;
	    auto img = Utility::ReadPNG<unsigned char, 1>(path);
	    ImageRef<InputLabel> ret(img.Width(), img.Height());
	    ret.Clear();

	    for( int y = 0; y < ret.Height(); ++y )
		for( int x = 0; x < ret.Width(); ++x )
		    ret(x,y)[0] = *(img.Ptr(x,y)) / (double) 255;
	    return ret;
	}

	ImageRef<InputLabel> LoadInputImageDLM(const std::string& path) const
	{
	    auto img = Utility::ReadDLM<float, 1>(path);
	    ImageRef<InputLabel> ret(img.Width(), img.Height());
	    ret.Clear();

	    for( int y = 0; y < ret.Height(); ++y )
		for( int x = 0; x < ret.Width(); ++x )
		    ret(x,y)[0] = *(img.Ptr(x,y));
	    return ret;
	}

	ImageRef<UnaryGroundLabel> LoadKernelDLM(const std::string& path) const
	{
	    auto img = Utility::ReadDLM<float, 1>(path);
	    ImageRef<UnaryGroundLabel> ret(img.Width(), img.Height());
	    ret.Clear();

	    for( int y = 0; y < ret.Height(); ++y )
		for( int x = 0; x < ret.Width(); ++x )
		    ret(x,y)[0] = *(img.Ptr(x,y));
	    return ret;
	}

	void GetInputAsITKImage(const ImageRef<InputLabel> input,
				ImageType::Pointer& Y) const
	{
	    const int cx = input.Width();
	    const int cy = input.Height();

	    ImageType::IndexType start;
	    start.Fill(0);

	    ImageType::SizeType size;
	    size[0] = cx;
	    size[1] = cy;

	    ImageType::RegionType region;
	    region.SetSize(size);
	    region.SetIndex(start);

	    Y->SetRegions(region);
	    Y->Allocate();

	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for(int y = 0; y < cy; ++y)
	    {
		for(int x = 0; x < cx; ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    Y->SetPixel(pixelIndex, input(x, y)[IDX_INPUT_IMAGE]);
		}
	    }
	}

	void GetKernelAsITKImage(const ImageRef<InputLabel> input, ImageType::Pointer& kernel) const
	{
	    int kerSz = GetKernelSize(input);

	    ImageType::IndexType start;
	    start.Fill(0);

	    ImageType::SizeType size;
	    size[0] = kerSz;
	    size[1] = kerSz;

	    ImageType::RegionType region;
	    region.SetSize(size);
	    region.SetIndex(start);

	    kernel->SetRegions(region);
	    kernel->Allocate();

	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for(int y = 0; y < kerSz; ++y)
	    {
		for(int x = 0; x < kerSz; ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    kernel->SetPixel(pixelIndex, input(x, y)[IDX_BLUR_KERNEL]);
		}
	    }
	}

	ImageRef<InputLabel> WriteITKKernelToInput(ImageRef<InputLabel> input,
						   const ImageType::Pointer& kernel) const
	{
	    // write kernel to IDX_BLUR_KERNEL channel
	    int kerSz = GetKernelSize(input);
	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for( int y = 0; y < kerSz; ++y )
	    {
		for( int x = 0; x < kerSz; ++x )
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    input(x,y)[IDX_BLUR_KERNEL] = kernel->GetPixel(pixelIndex);
		}
	    }

	    // enter blur kernel size as last argument to IDX_BLUR_KERNEL channel
	    const int cx = input.Width();
	    const int cy = input.Height();
	    assert(cx > kerSz && cy > kerSz);

	    input(cx-1, cy-1)[IDX_BLUR_KERNEL] = kerSz;

	    return input;
	}

	ImageRef<InputLabel>
	    PreProcessInput(ImageRef<InputLabel> input, std::string fileName,
			    ImageRefC<UnaryGroundLabel> previousPrediction = ImageRef<UnaryGroundLabel>(0,0)) const
	{
	    // Cascade level > 0
	    if( ! previousPrediction.Width() == 0 )
	    {
		const int cx = input.Width();
		const int cy = input.Height();

		// Set previous estimate of sharp image
#pragma omp parallel for
		for( int y = 0; y < cy; ++y )
		    for( int x = 0; x < cx; ++x )
			input(x,y)[IDX_SHARP_IMAGE] = previousPrediction(x,y)[0];

		if (interleaved)
		{
		    input = UpdateBlurKernel(input);
		}
	    }

	    // 2) Update the filter responses
	    return UpdateFilterResponses(input);
	}

	ImageRef<InputLabel>
	    UpdateBlurKernel(ImageRef<InputLabel> input) const
	{
	    const int kerSz = GetKernelSize(input);
	    const int offset = (kerSz-1)/2;
	    const int cx = input.Width();
	    const int cy = input.Height();

	    // extract sharp image estimate
	    ImageRef<InputLabel> tmp(cx, cy);
#pragma omp parallel for
	    for( int y = 0; y < cy; ++y )
		for( int x = 0; x < cx; ++x )
		    tmp(x,y)[IDX_INPUT_IMAGE] = input(x,y)[IDX_SHARP_IMAGE];
	    ImageType::Pointer I = ImageType::New();
	    GetInputAsITKImage(tmp, I);
	    CropSym(I, offset);

	    ImageType::Pointer y = ImageType::New();
	    GetInputAsITKImage(input, y);

	    CropSym(y, offset);
	    ImageType::SizeType sz = GetSize(y);
	    CropSym(y, offset);

	    ImageType::Pointer kernel = ImageType::New();
	    GetKernelAsITKImage(input, kernel);
	    kernel = Flip(kernel);

	    // latent image derivatives
	    ImageType::Pointer Mgrad0 = Gradz(I, 0);
	    ImageType::Pointer Mgrad1 = Gradz(I, 1);

	    // build r.h.s.
	    ImageType::Pointer b = ZeroImg(kerSz, kerSz);
	    AddImgs(b, ConvValid2(Mgrad0, Flip(Gradz(y, 0))));
	    AddImgs(b, ConvValid2(Mgrad1, Flip(Gradz(y, 1))));

	    // build part of preconditioner
	    ImageType::Pointer M0 = ZeroImg(kerSz, kerSz);
	    ImageType::Pointer aux = Ones(sz[0]-kerSz+1, sz[1]-kerSz+1);
	    AddImgs(M0, ConvValid2(PSquare(Mgrad0), aux));
	    AddImgs(M0, ConvValid2(PSquare(Mgrad1), aux));

	    double res; double gamma = 1;
	    ImageType::Pointer Sbar = Ones(kerSz, kerSz);
	    ImageType::Pointer pkernel = kernel;

	    while (true)
	    {
		ImageType::Pointer Psi = CwiseMax(kernel, 1E-5);
		ImageType::Pointer Diag = MultScal(DivImgs(Sbar,Psi), gamma);

		CGsolve(Diag, Mgrad0, Mgrad1, b, AddImgs2(M0, Diag), kernel);
		kernel = CwiseMax(kernel, 0);
		NormalizeKer(kernel);

		res = Norm(SubImgs(kernel, pkernel)) / Norm(pkernel);
		MyMonitor::Report("	Kernel iterate residual: %.2f\n", res);
		if (res <= 1E-1)
		    break;
		pkernel = kernel;
	    }
	    kernel = Flip(kernel);

	    return WriteITKKernelToInput(input, kernel);
	}

	static ImageRef<InputLabel>
	    UpdateFilterResponses(ImageRef<InputLabel> image)
	{
	    // Computes a 2D convolution with each filter and stores
	    // the response in the input image for later use in the
	    // linear basis and the feature checks
	    const int cx = image.Width();
	    const int cy = image.Height();

	    const int fcy_offset = Filterbank::filter_size_y/2;
	    const int fcx_offset = Filterbank::filter_size_x/2;

	    std::vector<FPType> filter_values(Filterbank::filter_size_y * Filterbank::filter_size_x * Filterbank::filter_count);
	    int idx = 0;
	    for (int fy = 0; fy < Filterbank::filter_size_y; ++fy)
		for (int fx = 0; fx < Filterbank::filter_size_x; ++fx)
		    for (int fi = 0; fi < Filterbank::filter_count; ++fi)
			filter_values[idx++] = (FPType) Filterbank::filter_values[fi][fy][fx];

#pragma omp parallel for
	    for (int y = 0; y < cy; ++y)
	    {
		for (int x = 0; x < cx; ++x)
		{
		    const FPType* filter_ptr = &(filter_values[0]);

		    auto &output_label = image(x,y);
		    // Initialize the filter responses to zero
		    memset(&output_label[IDX_FILTER_RESPONSES], 0, Filterbank::filter_count * sizeof(FPType));

		    for (int fy = 0; fy < Filterbank::filter_size_y; ++fy)
		    {
			int eff_y = y+fy-fcy_offset;
			eff_y = std::max(0, std::min(eff_y, cy-1));

			for (int fx = 0; fx < Filterbank::filter_size_x; ++fx)
			{
			    int eff_x = x+fx-fcx_offset;
			    eff_x = std::max(0, std::min(eff_x, cx-1));

			    const auto input = image(eff_x, eff_y)[IDX_SHARP_IMAGE];
			    FPType* output_ptr = &output_label[IDX_FILTER_RESPONSES];

			    for (int fi = 0; fi < Filterbank::filter_count; ++fi) {
				const auto filter_value = *filter_ptr++;
				*output_ptr++ += input * filter_value;
			    }
			}
		    }
		}
	    }
	    return image;
	}

	void Interleaved()
	{
	    interleaved = true;
	}

    private:

#ifdef USE_MPI
	void ReadDescriptor()
	{
	    const std::string dpath = path + "/" + type + ".txt";
	    std::ifstream ifs(dpath);

	    std::cerr << "reading " << dpath << std::endl;

	    if( ifs.fail() ) {
		std::cerr << "failed to open " << dpath << std::endl;
		throw IOException("failed to open '" + dpath + "'");
	    }

	    std::string file; size_t line = 0;
	    while(std::getline(ifs, file))
	    {
		if( (file != "") && (line++ % MPI::Communicator().size() == MPI::Communicator().rank()) ) {
		    fileNames.push_back(file);
		}
	    }
	    inputImages.resize(fileNames.size());
	    groundTruthImages.resize(fileNames.size());
	}
#else
	void ReadDescriptor()
	{
	    const std::string dpath = path + "/" + type + ".txt";
	    std::ifstream ifs(dpath);

	    if( ifs.fail() )
		throw IOException("failed to open '" + dpath + "'");

	    std::string file;
	    while(std::getline(ifs, file))
	    {
		if( file != "" )
		    fileNames.push_back(file);
	    }
	    groundTruthImages.resize(fileNames.size());
	    inputImages.resize(fileNames.size());
	}
#endif

	void SaveKernel(ImageRef<InputLabel> input, const std::string& path) const
	{
	    int kerSz = (int) input(input.Width()-1, input.Height()-1)[Dataset::IDX_BLUR_KERNEL];
	    ImageRef<float> img(kerSz, kerSz);
	    for( int y = 0; y < img.Height(); ++y )
		for( int x = 0; x < img.Width(); ++x )
		    *(img.Ptr(x,y)) = (float) input(x,y)[Dataset::IDX_BLUR_KERNEL];
	    Utility::WriteDLM(img, path);
	}

	const std::string path;
	const std::string type;

	bool interleaved;
	std::vector<std::string> fileNames;
	mutable std::vector<int> kernelSizes;
	mutable std::vector<ImageRef<InputLabel>> inputImages;
	mutable std::vector<ImageRefC<UnaryGroundLabel>> groundTruthImages;
    };

} // namespace Deblurring


#endif // _H_DATASET_H_
