#ifndef _H_BLUR_KERNEL_OPERATOR_H_
#define _H_BLUR_KERNEL_OPERATOR_H_

#include "RTF/Compute.h"
#include "RTF/LinearOperator.h"

#include "Dataset.h"
#include "Filters.h"

namespace Deblurring
{
    template<typename TFeature, typename TUnaryGroundLabel>
	class BlurKernelOperator : public LinearOperator::OperatorBase<TFeature, TUnaryGroundLabel, LinearOperator::DefaultWeights<typename TUnaryGroundLabel::ValueType>>
    {
    public:
	typedef LinearOperator::OperatorBase<TFeature, TUnaryGroundLabel, LinearOperator::DefaultWeights<typename TUnaryGroundLabel::ValueType>> Base;
	typedef typename TUnaryGroundLabel::ValueType TValue;
	typedef typename Base::SystemVectorRef SystemVectorRef;
	typedef typename Base::SystemVectorCRef SystemVectorCRef;

    BlurKernelOperator() : kernel(ImageType::New())
	{
	}

	virtual void AddInImplicitMatrixMultipliedBy(const typename TFeature::PreProcessType& prep,
						     const SystemVectorRef& Qr, const SystemVectorCRef& r) const
	{
	    // ----- Add (K^T K) r to the Qr vector
	    // Note: AddInLinearContribution is always called before

	    // transform r to ITK image format
	    ImageType::Pointer smd = ImageType::New();
	    GetCSysVecAsITK(r, smd);

	    // compute (K^T K) r
	    ConvValid(smd, kernel);
	    ConvFull(smd, Flip(kernel));

	    // add result to the Qr vector
	    ImageType::SizeType sz = GetSize(smd);
	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for(size_t y = 0; y < sz[1]; ++y)
	    {
		for(size_t x = 0; x < sz[0]; ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    Qr(x, y)[0] += smd->GetPixel(pixelIndex);
		}
	    }
	}

	virtual void AddInLinearContribution(const typename TFeature::PreProcessType& prep, const SystemVectorRef& l) const
	{
	    // ----- Add K^T y to the l vector

	    // kernel size: Last element of kernel channel
	    int kerSz = getKerSz(prep);
	    int offset = (kerSz-1)/2;

	    // store kernel in ITK format
	    GetKernelAsITK(prep);

	    // transform blurry image to ITK format for convolution
	    ImageType::Pointer Kty = ImageType::New();
	    GetInputImageAsITK(prep, Kty);

	    // compute K^T y
	    CropSym(Kty, offset);
	    ConvFull(Kty, Flip(kernel));

	    // Add result to the l vector
	    ImageType::SizeType size = GetSize(Kty);
	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for(size_t y = 0; y < size[1]; ++y)
	    {
		for(size_t x = 0; x < size[0]; ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    l(x, y)[0] += Kty->GetPixel(pixelIndex);
		}
	    }
	}

	virtual void AddInDiagonal(const typename TFeature::PreProcessType& prep, const SystemVectorRef& diag) const
	{
	    // cut out blur kernel
	    int kerSz = getKerSz(prep);
	    int offset = (kerSz-1)/2;

	    // compute diagonal
	    ImageType::Pointer out = Ones(prep.Width() - kerSz + 1, prep.Height() - kerSz + 1);
	    ZeroPad(out, offset);
	    ConvSameZeroBnd(out, PSquare(Flip(kernel)));

	    // add result to the diag vector
	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for(size_t y = 0; y < prep.Height(); ++y)
	    {
		for(size_t x = 0; x < prep.Width(); ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    diag(x, y)[0] += out->GetPixel(pixelIndex);
		}
	    }
	}

	// Boilerplate code

	virtual void AccumulateGradient(const typename TFeature::PreProcessType& prep,
					const SystemVectorCRef& muLeftRef, const SystemVectorCRef& muRightRef, TValue normC) const
	{
	}

	const TValue* CheckFeasibility(const TValue *ws, bool& feasible) const
	{
	    feasible = true;
	    return ws;
	}

	TValue* Project(TValue *ws) const
	{
	    return ws;
	}

	size_t NumWeights() const
	{
	    return 0;
	}

	const TValue* SetWeights(const TValue *ws)
	{
	    return ws;
	}

	TValue* GetWeights(TValue *ws) const
	{
	    return ws;
	}

	TValue* GetGradientAddPrior(TValue *gs, TValue& objective) const
	{
	    return gs;
	}

	void ClearGradient()
	{
	}

	void ResetWeights()
	{
	}

	void Print() const
	{
	}

	int Type() const
	{
	    return TYPE_BLUR_KERNEL_OPERATOR;
	}

	std::istream& Deserialize(std::istream& in)
	{
	    return in;
	}

	std::ostream& Serialize(std::ostream& out) const
	{
	    return out;
	}

#ifdef _OPENMP
	// We don't need a lock
	void InitializeLocks()
	{
	}

	void DestroyLocks()
	{
	}
#endif

    private:
	const ImageType::Pointer kernel;

	void GetCSysVecAsITK(const SystemVectorCRef& r, const ImageType::Pointer& R) const
	{
	    ImageType::IndexType start;
	    start.Fill(0);

	    ImageType::SizeType size;
	    size[0] = r.Width();
	    size[1] = r.Height();

	    ImageType::RegionType region;
	    region.SetSize(size);
	    region.SetIndex(start);

	    R->SetRegions(region);
	    R->Allocate();

	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for(size_t y = 0; y < size[1]; ++y)
	    {
		for(size_t x = 0; x < size[0]; ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    R->SetPixel(pixelIndex, r(x,y)[0]);
		}
	    }
	}

	void GetInputImageAsITK(const typename TFeature::PreProcessType& prep,
				const ImageType::Pointer& Y) const
	{
	    ImageType::IndexType start;
	    start.Fill(0);

	    ImageType::SizeType size;
	    size[0] = prep.Width();
	    size[1] = prep.Height();

	    ImageType::RegionType region;
	    region.SetSize(size);
	    region.SetIndex(start);

	    Y->SetRegions(region);
	    Y->Allocate();

	    ImageType::IndexType pixelIndex;
#pragma omp parallel for
	    for(size_t y = 0; y < prep.Height(); ++y)
	    {
		for(size_t x = 0; x < prep.Width(); ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    Y->SetPixel(pixelIndex, prep(x, y)[Dataset::IDX_INPUT_IMAGE]);
		}
	    }
	}

	void GetKernelAsITK(const typename TFeature::PreProcessType& prep) const
	{
	    int kerSz = getKerSz(prep);

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
	    for(size_t y = 0; y < kerSz; ++y)
	    {
		for(size_t x = 0; x < kerSz; ++x)
		{
		    pixelIndex[0] = x;
		    pixelIndex[1] = y;
		    kernel->SetPixel(pixelIndex, prep(x, y)[Dataset::IDX_BLUR_KERNEL]);
		}
	    }
	}

	inline int getKerSz(const typename TFeature::PreProcessType& prep) const
	{
	    return (int) prep(prep.Width()-1, prep.Height()-1)[Dataset::IDX_BLUR_KERNEL];
	}
    };
}

#endif // _H_BLUR_KERNEL_OPERATOR_H_
