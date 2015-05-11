#ifndef _H_FILTERS_H_
#define _H_FILTERS_H_

#include "RTF/Monitor.h"
#include <itkFlipImageFilter.h>
#include <itkFFTConvolutionImageFilter.h>
#include <itkZeroFluxNeumannPadImageFilter.h>
#include <itkSquareImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkChangeInformationImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkAccumulateImageFilter.h>
#include <itkSubtractImageFilter.h>

#if defined(USE_FFTWF) || defined(USE_FFTWD)
#include <itkFFTWGlobalConfiguration.h>
#endif

typedef Monitor::DefaultMonitor MyMonitor;
typedef double PixelType;
typedef itk::Image<PixelType, 2> ImageType;
typedef itk::ExtractImageFilter< ImageType, ImageType > ExtractImageFilterType;
typedef itk::FFTConvolutionImageFilter<ImageType> FFTConvolutionImageFilterType;
typedef itk::ChangeInformationImageFilter< ImageType > ChangeInformationImageFilterType;
typedef itk::ConstantPadImageFilter <ImageType, ImageType> ConstantPadImageFilterType;
typedef itk::FlipImageFilter<ImageType> FlipImageFilterType;
typedef itk::SquareImageFilter<ImageType, ImageType> SquareImageFilterType;
typedef itk::ZeroFluxNeumannPadImageFilter<ImageType, ImageType > ZeroFluxNeumannPadImageFilterType;
typedef itk::AddImageFilter <ImageType, ImageType > AddImageFilterType;
typedef itk::MultiplyImageFilter <ImageType, ImageType > MultiplyImageFilterType;
typedef itk::BinaryThresholdImageFilter <ImageType, ImageType > BinaryThresholdImageFilterType;
typedef itk::AccumulateImageFilter <ImageType, ImageType> AccumulateImageFilterType;
typedef itk::ImageRegionIterator< ImageType > IteratorType;

void ShiftIndex(ImageType::Pointer& image, int shift0, int shift1)
{
    ChangeInformationImageFilterType::Pointer indexChangeFilter = ChangeInformationImageFilterType::New();
    indexChangeFilter->ChangeRegionOn();
    ChangeInformationImageFilterType::OutputImageOffsetValueType indexShift[2];
    indexShift[0] = shift0;
    indexShift[1] = shift1;
    indexChangeFilter->SetOutputOffset( indexShift );
    indexChangeFilter->SetInput( image );
    indexChangeFilter->UpdateLargestPossibleRegion();

    image = indexChangeFilter->GetOutput();
}

ImageType::SizeType GetSize(const ImageType::Pointer& image)
{
    ImageType::RegionType region = image->GetLargestPossibleRegion();
    return region.GetSize();
}

void Crop(ImageType::Pointer& image, const ImageType::IndexType& start,
	  const ImageType::SizeType& sz)
{
    // crop out part of an image
    ExtractImageFilterType::Pointer extracter = ExtractImageFilterType::New();
    extracter->SetDirectionCollapseToIdentity();
    ImageType::RegionType region(start, sz);
    extracter->SetExtractionRegion(region);
    extracter->SetInput(image);
    extracter->UpdateLargestPossibleRegion();
    image = extracter->GetOutput();
    ShiftIndex(image, -start[0], -start[1]);
}

void CropSym(ImageType::Pointer& image, int offset)
{
    // crop out interior of an image with symmetric padding on all sides
    ImageType::SizeType sz = GetSize(image);
    sz[0] = sz[0] - 2*offset;
    sz[1] = sz[1] - 2*offset;
    ImageType::IndexType start;
    start[0] = offset; start[1] = offset;

    Crop(image, start, sz);
}

void RemoveLast(ImageType::Pointer& image)
{
    // removes the last row and column of an image
    ImageType::SizeType imSz = GetSize(image);
    ImageType::SizeType newSz;
    newSz[0] = imSz[0]-1;
    newSz[1] = imSz[1]-1;

    ImageType::IndexType start;
    start.Fill(0);

    ExtractImageFilterType::Pointer extracter = ExtractImageFilterType::New();
    extracter->SetDirectionCollapseToIdentity();
    ImageType::RegionType newRegion(start, newSz);
    extracter->SetExtractionRegion(newRegion);
    extracter->SetInput(image);
    extracter->UpdateLargestPossibleRegion();
    image = extracter->GetOutput();
}

ImageType::Pointer Flip(const ImageType::Pointer& image)
{
    // flips a blur kernel
    FlipImageFilterType::Pointer flipper = FlipImageFilterType::New();

    itk::FixedArray<bool, 2> flipAxes;
    flipAxes[0] = true;
    flipAxes[1] = true;
    flipper->SetFlipAxes(flipAxes);
    flipper->SetInput(image);

    ChangeInformationImageFilterType::Pointer indexChangeFilter = ChangeInformationImageFilterType::New();
    indexChangeFilter->SetInput( flipper->GetOutput() );
    indexChangeFilter->ChangeOriginOn();
    ImageType::PointType origin; origin[0] = 0; origin[1] = 0;
    indexChangeFilter->SetOutputOrigin(origin);
    indexChangeFilter->UpdateLargestPossibleRegion();

    return indexChangeFilter->GetOutput();
}

void ZeroPad(ImageType::Pointer& image, int width)
{
    ConstantPadImageFilterType::Pointer zeroPadder = ConstantPadImageFilterType::New();
    zeroPadder->SetConstant(0);
    zeroPadder->SetInput(image);

    ImageType::SizeType bound;
    bound[0] = width;
    bound[1] = width;

    zeroPadder->SetPadBound(bound);
    zeroPadder->UpdateLargestPossibleRegion();

    image = zeroPadder->GetOutput();
    ShiftIndex(image, width, width);
}

void ZeroPadAsym(ImageType::Pointer& image, int width0, int width1)
{
    ConstantPadImageFilterType::Pointer asymZeroPadder = ConstantPadImageFilterType::New();
    asymZeroPadder->SetConstant(0);
    asymZeroPadder->SetInput(image);

    ImageType::SizeType lowerExtendRegion;
    lowerExtendRegion[0] = width0;
    lowerExtendRegion[1] = width1;

    ImageType::SizeType upperExtendRegion;
    upperExtendRegion[0] = width0;
    upperExtendRegion[1] = width1;

    asymZeroPadder->SetPadLowerBound(lowerExtendRegion);
    asymZeroPadder->SetPadUpperBound(upperExtendRegion);
    asymZeroPadder->UpdateLargestPossibleRegion();

    image = asymZeroPadder->GetOutput();
    ShiftIndex(image, width0, width1);
}

void NeumannPad(ImageType::Pointer& image, int width)
{
    // pad image by replicating the boundaries
    ZeroFluxNeumannPadImageFilterType::Pointer neumannPadder = ZeroFluxNeumannPadImageFilterType::New();
    neumannPadder->SetInput(image);

    ImageType::SizeType bound; bound[0] = width; bound[1] = width;
    neumannPadder->SetPadBound(bound);
    neumannPadder->UpdateLargestPossibleRegion();

    image = neumannPadder->GetOutput();
    ShiftIndex(image, width, width);
}

ImageType::Pointer ZeroImg(int sz0, int sz1)
{
    // create a zero image of specified dimension
    ImageType::Pointer zimage = ImageType::New();
    ImageType::IndexType start;
    start.Fill(0);

    ImageType::SizeType size;
    size[0] = sz0;
    size[1] = sz1;

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    zimage->SetRegions(region);
    zimage->Allocate();
    zimage->FillBuffer(0);

    return zimage;
}

ImageType::Pointer Ones(int sz0, int sz1)
{
    // create an image of ones of specified dimension
    ImageType::Pointer oimage = ImageType::New();
    ImageType::IndexType start;
    start.Fill(0);

    ImageType::SizeType size;
    size[0] = sz0;
    size[1] = sz1;

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    oimage->SetRegions(region);
    oimage->Allocate();
    oimage->FillBuffer(1);

    return oimage;
}

ImageType::Pointer PSquare(const ImageType::Pointer& image)
{
    SquareImageFilterType::Pointer squareFilter = SquareImageFilterType::New();

    squareFilter->SetInput(image);
    squareFilter->UpdateLargestPossibleRegion();
    return squareFilter->GetOutput();
}

void ConvValid(ImageType::Pointer& image, const ImageType::Pointer& kernel)
{
    ImageType::SizeType sz = GetSize(kernel);

    FFTConvolutionImageFilterType::Pointer validConvFilter = FFTConvolutionImageFilterType::New();
    validConvFilter->SetOutputRegionModeToValid();
    validConvFilter->SetKernelImage( kernel );
    validConvFilter->SetInput( image );
    validConvFilter->UpdateLargestPossibleRegion();

    int offset0 = (sz[0]-1)/2; int offset1 = (sz[1]-1)/2;
    image = validConvFilter->GetOutput();
    ShiftIndex(image, -offset0, -offset1);
}

ImageType::Pointer ConvValid2(const ImageType::Pointer& image, const ImageType::Pointer& kernel)
{
    ImageType::SizeType sz = GetSize(kernel);

    FFTConvolutionImageFilterType::Pointer validConvFilter = FFTConvolutionImageFilterType::New();
    validConvFilter->SetOutputRegionModeToValid();
    validConvFilter->SetKernelImage( kernel );
    validConvFilter->SetInput( image );
    validConvFilter->UpdateLargestPossibleRegion();

    int offset0 = (sz[0]-1)/2; int offset1 = (sz[1]-1)/2;
    ImageType::Pointer output = ImageType::New();
    output = validConvFilter->GetOutput();
    ShiftIndex(output, -offset0, -offset1);
    return output;
}

void ConvSameZeroBnd(ImageType::Pointer& image, const ImageType::Pointer& kernel)
{
    ImageType::SizeType sz = GetSize(kernel);
    int offset0 = (sz[0]-1)/2; int offset1 = (sz[1]-1)/2;
    ZeroPadAsym(image, offset0, offset1);

    ConvValid(image, kernel);
}

void ConvFull(ImageType::Pointer& image, const ImageType::Pointer& kernel)
{
    // Note: This code uses zero values outside the image boundaries
    ImageType::SizeType sz = GetSize(kernel);
    FFTConvolutionImageFilterType::Pointer convFilter = FFTConvolutionImageFilterType::New();
    convFilter->SetKernelImage( kernel );

    // Pad the image with zeros to achieve full convolution
    ZeroPad(image, ceil(((double)sz[0]-1)/2));

    convFilter->SetInput(image);
    convFilter->UpdateLargestPossibleRegion();

    image = convFilter->GetOutput();
    if (sz[0]%2 == 0)
    {
	RemoveLast(image);
    }
}

ImageType::Pointer GradzFilter(int direction)
{
    // compute the derivative kernel
    ImageType::Pointer kernel = ImageType::New();
    ImageType::IndexType start;
    start.Fill(0);

    ImageType::SizeType size;
    size.Fill(3);

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    kernel->SetRegions(region);
    kernel->Allocate();
    kernel->FillBuffer(0);

    ImageType::IndexType index;
    index[0] = 1;
    index[1] = 1;
    kernel->SetPixel( index, 1 );
    if (direction==0)
    {
	index[0] = 2;
	kernel->SetPixel( index, -1 );
    }
    else
    {
	index[1] = 2;
	kernel->SetPixel( index, -1 );
    }

    return kernel;
}

ImageType::Pointer ConvSame(const ImageType::Pointer& image, const ImageType::Pointer& kernel)
{
    ImageType::SizeType sz = GetSize(kernel);
    FFTConvolutionImageFilterType::Pointer convolutionFilter;
    convolutionFilter = FFTConvolutionImageFilterType::New();
    convolutionFilter->SetKernelImage( kernel );
    convolutionFilter->SetInput(image);
    convolutionFilter->UpdateLargestPossibleRegion();
    return convolutionFilter->GetOutput();
}

ImageType::Pointer Gradz(const ImageType::Pointer& image, int direction)
{
    // returns image gradients
    ImageType::Pointer kernel = GradzFilter(direction);
    return ConvSame(image, kernel);
}

void AddImgs(ImageType::Pointer& image1, const ImageType::Pointer& image2)
{
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    addFilter->SetInput1(image1);
    addFilter->SetInput2(image2);
    addFilter->UpdateLargestPossibleRegion();
    image1 = addFilter->GetOutput();
}

ImageType::Pointer AddImgs2(const ImageType::Pointer& image1, const ImageType::Pointer& image2)
{
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    addFilter->SetInput1(image1);
    addFilter->SetInput2(image2);
    addFilter->UpdateLargestPossibleRegion();
    return addFilter->GetOutput();
}

ImageType::Pointer MultImgs(const ImageType::Pointer& image1, const ImageType::Pointer& image2)
{
    MultiplyImageFilterType::Pointer multiplyFilter = MultiplyImageFilterType::New ();
    multiplyFilter->SetInput1(image1);
    multiplyFilter->SetInput2(image2);
    multiplyFilter->UpdateLargestPossibleRegion();
    return multiplyFilter->GetOutput();
}

ImageType::Pointer GreaterIndex(const ImageType::Pointer& image, const double value)
{
    //returns a binary map for image pixels greater than a value
    BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
    thresholdFilter->SetInput(image);
    thresholdFilter->SetLowerThreshold(-std::numeric_limits<double>::infinity());
    thresholdFilter->SetUpperThreshold(value);
    thresholdFilter->SetInsideValue(0);
    thresholdFilter->SetOutsideValue(1);
    thresholdFilter->UpdateLargestPossibleRegion();
    return thresholdFilter->GetOutput();
}

ImageType::Pointer AddScal(const ImageType::Pointer& image, const double scalar)
{
    typedef itk::AddImageFilter <ImageType, ImageType > AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New ();
    addFilter->SetInput1(image);
    addFilter->SetConstant2(scalar);
    addFilter->UpdateLargestPossibleRegion();
    return addFilter->GetOutput();
}

ImageType::Pointer MultScal(const ImageType::Pointer& image, const double scalar)
{
    MultiplyImageFilterType::Pointer multiplyImageFilter = MultiplyImageFilterType::New ();
    multiplyImageFilter->SetInput(image);
    multiplyImageFilter->SetConstant(scalar);
    multiplyImageFilter->UpdateLargestPossibleRegion();
    return multiplyImageFilter->GetOutput();
}

ImageType::Pointer InvMap(ImageType::Pointer& map)
{
    //invert binary pixel map
    return AddScal(MultScal(map, -1), 1);
}

ImageType::Pointer CwiseMax(const ImageType::Pointer& image, const double value)
{
    //returns the component wise maximum
    ImageType::Pointer index = GreaterIndex(image, value);
    return AddImgs2(MultImgs(image, index), MultScal(InvMap(index), value));
}

ImageType::Pointer DivImgs(const ImageType::Pointer& image1, const ImageType::Pointer& image2)
{
    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideImageFilterType;
    DivideImageFilterType::Pointer divideImageFilter = DivideImageFilterType::New();
    divideImageFilter->SetInput1(image1);
    divideImageFilter->SetInput2(image2);
    divideImageFilter->UpdateLargestPossibleRegion();
    return divideImageFilter->GetOutput();
}

ImageType::Pointer SubImgs(const ImageType::Pointer& image1, const ImageType::Pointer& image2)
{
    typedef itk::SubtractImageFilter <ImageType, ImageType > SubtractImageFilterType;
    SubtractImageFilterType::Pointer subFilter = SubtractImageFilterType::New ();
    subFilter->SetInput1(image1);
    subFilter->SetInput2(image2);
    subFilter->UpdateLargestPossibleRegion();
    return subFilter->GetOutput();
}

ImageType::Pointer SumOverDim(const ImageType::Pointer& image, const int dim)
{
    AccumulateImageFilterType::Pointer accum = AccumulateImageFilterType::New();
    accum->SetInput(image);
    accum->SetAccumulateDimension(dim);
    accum->UpdateLargestPossibleRegion();
    return accum->GetOutput();
}

double SumOverImg(const ImageType::Pointer& image)
{
    ImageType::Pointer sum = SumOverDim(SumOverDim(image, 0), 1);
    ImageType::IndexType pixelIndex;
    pixelIndex[0] = 0;
    pixelIndex[1] = 0;
    return sum->GetPixel(pixelIndex);
}

double Norm(const ImageType::Pointer& image)
{
    return sqrt(SumOverImg(PSquare(image)));
}

ImageType::Pointer DivScal(const ImageType::Pointer& image, const double scalar)
{
    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideImageFilterType;
    DivideImageFilterType::Pointer divideImageFilter = DivideImageFilterType::New();
    divideImageFilter->SetInput(image);
    divideImageFilter->SetConstant(scalar);
    divideImageFilter->UpdateLargestPossibleRegion();
    return divideImageFilter->GetOutput();
}

void NormalizeKer(ImageType::Pointer& kernel)
{
    kernel = DivScal(kernel, SumOverImg(kernel));
}

ImageType::Pointer AuxMat(const ImageType::Pointer& Diag, const ImageType::Pointer& Mgrad0,
			  const ImageType::Pointer& Mgrad1, const ImageType::Pointer& kernel)
{
    ImageType::SizeType sz = GetSize(kernel);
    ImageType::Pointer Out = ZeroImg(sz[0], sz[1]);

    AddImgs(Out, ConvValid2(Mgrad0, Flip(ConvValid2(Mgrad0, Flip(kernel)))));
    AddImgs(Out, ConvValid2(Mgrad1, Flip(ConvValid2(Mgrad1, Flip(kernel)))));
    AddImgs(Out, MultImgs(kernel, Diag));
    return Out;
}

void CGsolve(const ImageType::Pointer& Diag, const ImageType::Pointer& Mgrad0, const ImageType::Pointer& Mgrad1,
	     const ImageType::Pointer& b, const ImageType::Pointer& M, ImageType::Pointer& kernel,
	     double tol = 1E-4, int maxitr = 200)
{
    ImageType::Pointer r0 = SubImgs(b, AuxMat(Diag, Mgrad0, Mgrad1, kernel));
    ImageType::Pointer z0 = DivImgs(r0, M);
    ImageType::Pointer p = z0;
    double res = Norm(r0);
    MyMonitor::Report("	PCG (kernel update): Initially: ||r|| %.4f\n", res);
    if (res<tol)
    {
	return;
    }

    double alpha, beta;
    ImageType::Pointer r1, z1;
    int i = 1;
    while(true)
    {
	alpha = SumOverImg(MultImgs(r0, z0)) / SumOverImg(MultImgs(AuxMat(Diag, Mgrad0, Mgrad1, p), p));
	AddImgs(kernel, MultScal(p, alpha));

	r1 = SubImgs(r0, MultScal(AuxMat(Diag, Mgrad0, Mgrad1, p), alpha));
	res = Norm(r1);

	if (res<tol)
	{
	    MyMonitor::Report("	PCG (kernel update): FinIt %3d: ||r|| %.4f\n", i, tol);
	    break;
	}

	z1 = DivImgs(r1, M);
	beta = SumOverImg(MultImgs(z1, r1)) / SumOverImg(MultImgs(z0, r0));
	p = AddImgs2(z1, MultScal(p, beta));

	r0 = r1; z0 = z1; ++i;
	if (i > maxitr)
	{
	    MyMonitor::Report("	PCG (kernel update): MaxIt %3d: ||r|| %.4f\n", i, res);
	    break;
	}
    }

}

#endif /* _H_FILTERS_H_ */
