#pragma once

#pragma managed(push, off)

#define RTF_NO_WIN32
#ifndef RTF_NO_WIN32
#include <atlbase.h>
#include <atltypes.h>
#endif
#include <omp.h>
#include <memory>
#include <cstring>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

class Image
{
public:
    Image() : m_pBits(0), m_pFree(0), m_nWidth(0), m_nHeight(0), m_nBands(0), m_nStrideBytes(0), m_nElementBytes(0)
    {
    }

    Image(const Image& obj) : m_pBits(0), m_pFree(0), m_nWidth(0), m_nHeight(0), m_nBands(0), m_nStrideBytes(0), m_nElementBytes(0)
    {
        Create(obj.Width(), obj.Height(), obj.ElementBytes(), obj.Bands());
    }

    virtual ~Image()
    {
        Destroy();
    }

    void Destroy()
    {
        delete [] m_pFree;
        m_pFree = m_pBits = 0;
        m_nStrideBytes = m_nElementBytes = m_nBands = m_nWidth = m_nHeight = 0;
    }

    void Create(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned int nRowAlignBytes = 4)
    {
        if(nWidth == m_nWidth && m_nHeight == nHeight && m_nElementBytes == nElementBytes && m_nBands == nBands)
            return;

        Destroy();
        size_t nPixelBytes  = nElementBytes * nBands;
        size_t nImgRowBytes = nPixelBytes * nWidth;
        size_t nStrideBytes = (nImgRowBytes + nRowAlignBytes - 1) & ~(nRowAlignBytes - 1);
        size_t nImgBytes = nStrideBytes * nHeight + nRowAlignBytes;

        m_pFree = new unsigned char [nImgBytes];
        m_pBits = m_pFree + nRowAlignBytes - ((std::ptrdiff_t)(m_pFree) & (nRowAlignBytes - 1));
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
    }

    void Attach(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned char* pBits, size_t nStrideBytes)
    {
        Destroy();
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_pBits = pBits;
        m_pFree = 0;
    }

    void Reference(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned char* pBits, size_t nStrideBytes)
    {
        Destroy();
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_pBits = pBits;
        m_pFree = 0;
    }

    void AttachData(int nWidth, int nHeight, size_t nElementBytes, unsigned int nBands, unsigned char* pBits, size_t nStrideBytes)
    {
        Destroy();
        m_nWidth = nWidth;
        m_nHeight = nHeight;
        m_nBands = nBands;
        m_nStrideBytes = nStrideBytes;
        m_nElementBytes = nElementBytes;
        m_pFree = m_pBits = pBits;
    }

    unsigned char* DetachData()
    {
        unsigned char* rv = m_pFree;
        m_pFree = 0;
        return rv;
    }

    void Clear()
    {
        std::memset(m_pBits, 0, (m_nHeight - 1) * m_nStrideBytes);
        std::memset(m_pBits + (m_nHeight - 1) * m_nStrideBytes, 0, m_nWidth * m_nBands * m_nElementBytes);
    }

    int Width() const
    {
        return m_nWidth;
    }

    int Height() const
    {
        return m_nHeight;
    }
#ifndef RTF_NO_WIN32
    CRect Rect() const
    {
        return CRect(0, 0, Width(), Height());
    }
    CSize Size() const
    {
        return CSize(Width(), Height());
    }
#endif
    size_t ElementBytes() const
    {
        return m_nElementBytes;
    }
    size_t StrideBytes() const
    {
        return m_nStrideBytes;
    }
    size_t PixelBytes() const
    {
        return m_nBands * m_nElementBytes;
    }
    unsigned int Bands() const
    {
        return m_nBands;
    }
    size_t Bpp() const
    {
        return PixelBytes() * 8;
    }

    unsigned char* BytePtr()
    {
        return m_pBits;
    }
    const unsigned char* BytePtr() const
    {
        return m_pBits;
    }

protected:
    unsigned char* m_pFree;
    unsigned char* m_pBits;
    int m_nWidth, m_nHeight;
    unsigned int m_nBands;
    size_t m_nElementBytes, m_nStrideBytes;
};

template <typename T, unsigned int nBands = 1>
class ImageT : public Image
{
public:
    ImageT()
    {
    }

    ImageT(int nWidth, int nHeight)
    {
        Create(nWidth, nHeight);
    }

    ImageT(int nWidth, int nHeight, unsigned int nRowAlignBytes)
    {
        Create(nWidth, nHeight, nRowAlignBytes);
    }

    ImageT(int nWidth, int nHeight, T* pBits, size_t nStride)
    {
        AttachData(nWidth, nHeight, SizeofT(), nBands, (unsigned char*)pBits, nStride);
    }

    void Create(int nWidth, int nHeight, unsigned int nRowAlignBytes = 4)
    {
        Image::Create(nWidth, nHeight, SizeofT(), nBands, nRowAlignBytes);
    }

    T* Ptr()
    {
        return (T*)BytePtr();
    }
    T* Ptr(int y)
    {
        return (T*)(BytePtr() + y * StrideBytes());
    }
    T* Ptr(int x, int y)
    {
        return (T*)(BytePtr() + y * StrideBytes() + x * PixelBytes());
    }

    const T* Ptr() const
    {
        return (const T*)BytePtr();
    }
    const T* Ptr(int y) const
    {
        return (const T*)(BytePtr() + y * StrideBytes());
    }
    const T* Ptr(int x, int y) const
    {
        return (const T*)(BytePtr() + y * StrideBytes() + x * PixelBytes());
    }

    T& operator()(int x, int y)
    {
        return *Ptr(x, y);
    }

    const T& operator()(int x, int y) const
    {
        return *Ptr(x, y);
    }

    size_t Stride() const
    {
        return StrideBytes() / SizeofT();
    }

    void Clear()
    {
        Image::Clear();
    }

    void Clear(const T& t)
    {
        for(int y = 0; y < Height(); y++)
        {
            int cx = Width() * Bands();
            T* p = Ptr(y);

            for(int x = 0; x < cx; x++)
                p[x] = t;
        }
    }

protected:
    static size_t SizeofT()
    {
        return (size_t)((T*)0 + 1);
    }
};

template <typename T, unsigned int nBands = 1>
class ImageRef
{
public:
    ImageRef() {}
    ImageRef(int nWidth, int nHeight) : m_p(new ImageT<T, nBands>(nWidth, nHeight))
    {
        m_p->Clear();
    }
    ImageRef(const ImageRef& rhs) : m_p(rhs.m_p) {}

    T* Ptr() const
    {
        return m_p->Ptr();
    }
    T* Ptr(int y) const
    {
        return m_p->Ptr(y);
    }
    T* Ptr(int x, int y) const
    {
        return m_p->Ptr(x, y);
    }

    ImageT<T, nBands>& Get() const
    {
        return *m_p;
    }

    int Width() const
    {
        return m_p->Width();
    }
    int Height() const
    {
        return m_p->Height();
    }
    unsigned int Bands() const
    {
        return nBands;
    }
    size_t StrideBytes() const
    {
        return m_p->StrideBytes();
    }
    void Clear() const
    {
        return m_p->Clear();
    }
    void Clear(const T& t) const
    {
        m_p->Clear(t);
    }
    std::shared_ptr<ImageT<T, nBands>> Ref() const
    {
        return m_p;
    }

    T& operator()(int x, int y) const
    {
        return *Ptr(x, y);
    }
    operator bool() const
    {
        return static_cast<bool>(m_p);
    }
protected:
    std::shared_ptr<ImageT<T, nBands>> m_p;
};

template <typename T, unsigned int nBands = 1>
class ImageRefC
{
public:
    ImageRefC() {}
    ImageRefC(const ImageRefC& rhs) : m_p(rhs.m_p) {}
    ImageRefC(const ImageRef<T, nBands>& rhs) : m_p(rhs.Ref()) {}

    const T* Ptr() const
    {
        return m_p->Ptr();
    }
    const T* Ptr(int y) const
    {
        return m_p->Ptr(y);
    }
    const T* Ptr(int x, int y) const
    {
        return m_p->Ptr(x, y);
    }

    int Width() const
    {
        return m_p->Width();
    }
    int Height() const
    {
        return m_p->Height();
    }
    unsigned int Bands() const
    {
        return nBands;
    }
    size_t StrideBytes() const
    {
        return m_p->StrideBytes();
    }

    const T& operator()(int x, int y) const
    {
        return *Ptr(x, y);
    }
    operator bool() const
    {
        return m_p ? true : false;
    }
protected:
    std::shared_ptr<const ImageT<T, nBands>> m_p;
};

#ifndef RTF_NO_WIN32
namespace Arithmetic
{

    template <typename TDst, typename TLhs, typename TOp>
    void UnaryPointwiseOperator(ImageT<TDst>& dst, const ImageT<TLhs>& lhs, TOp& op, const CRect* prDst = 0, const CPoint* pptLhs = 0, TLhs tExt = (TLhs)0)
    {
        CRect rDArg(prDst ? *prDst : dst.Rect());
        CPoint ptLArg(pptLhs ? *pptLhs : CPoint(0, 0));
        CRect rDst(rDArg & dst.Rect());
        CSize szLOffset(ptLArg - rDArg.TopLeft());
        CRect rLWhole(lhs.Rect());
        TLhs L;
        const TLhs* pL;
        TDst* pD;
        CRect rCheck(rDst + szLOffset);

        if((rCheck & rLWhole) == rCheck)
        {
            // Loop without needing to check bounds
            #pragma omp parallel for
            for(int y = rDst.top; y < rDst.bottom; y++)
            {
                pL = lhs.Ptr(szLOffset.cx, szLOffset.cy + y);
                pD = dst.Ptr(y);

                for(int x = rDst.left; x < rDst.right; x++)
                    pD[x] = op(pL[x]);
            }
        }
        else
        {
            // Loop with a bounds check
            #pragma omp parallel for
            for(int y = rDst.top; y < rDst.bottom; y++)
            {
                pL = lhs.Ptr(szLOffset.cx, szLOffset.cy + y);
                pD = dst.Ptr(y);

                for(int x = rDst.left; x < rDst.right; x++)
                {
                    L = rLWhole.PtInRect(CPoint(x, y) + szLOffset) ? pL[x] : tExt;
                    pD[x] = op(L);
                }
            }
        }
    }

    template <typename TDst, typename TLhs, typename TRhs, typename TOp>
    inline void BinaryPointwiseOperator(
        ImageT<TDst>& dst,
        const ImageT<TLhs>& lhs,
        const ImageT<TRhs>& rhs,
        TOp& op,
        const CRect* prDst = NULL,
        const CPoint* pptLhs = NULL,
        const CPoint* pptRhs = NULL
    )
    {
        CRect rDArg(prDst ? *prDst : dst.Rect());
        CPoint ptLArg(pptLhs ? *pptLhs : CPoint(0, 0));
        CPoint ptRArg(pptRhs ? *pptRhs : CPoint(0, 0));
        CRect rDst(rDArg & dst.Rect());
        CSize szLOffset(ptLArg - rDArg.TopLeft());
        CSize szROffset(ptRArg - rDArg.TopLeft());
        CRect rLWhole(lhs.Rect());
        CRect rRWhole(rhs.Rect());
        TLhs L;
        TRhs R;
        const TLhs* pL;
        const TRhs* pR;
        TDst* pD;
        CRect rCheckL(rDst + szLOffset);
        CRect rCheckR(rDst + szROffset);

        if(((rCheckL & rLWhole) == rCheckL) && ((rCheckR & rRWhole) == rCheckR))
        {
            #pragma omp parallel for

            for(int y = rDst.top; y < rDst.bottom; y++)
            {
                pD = dst.Ptr(y);
                pL = lhs.Ptr(szLOffset.cx, y + szLOffset.cy);
                pR = rhs.Ptr(szROffset.cx, y + szROffset.cy);

                for(int x = rDst.left; x < rDst.right; x++)
                    pD[x] = op(pL[x], pR[x]);
            }
        }
        else
        {
            #pragma omp parallel for

            for(int y = rDst.top; y < rDst.bottom; y++)
            {
                pD = dst.Ptr(y);
                pL = lhs.Ptr(szLOffset.cx, y + szLOffset.cy);
                pR = rhs.Ptr(szROffset.cx, y + szROffset.cy);

                for(int x = rDst.left; x < rDst.right; x++)
                {
                    L = rLWhole.PtInRect(CPoint(x, y) + szLOffset) ? pL[x] : (TLhs)0;
                    R = rRWhole.PtInRect(CPoint(x, y) + szROffset) ? pR[x] : (TRhs)0;
                    pD[x] = op(L, R);
                }
            }
        }
    }

    template <typename TDst, typename TIn1, typename TIn2, typename TIn3, typename TOp>
    inline void TertiaryPointwiseOperator(
        ImageT<TDst>& dst,
        const ImageT<TIn1>& in1,
        const ImageT<TIn2>& in2,
        const ImageT<TIn3>& in3,
        TOp& op
    )
    {
        CRect rDst(dst.Rect());
        rDst &= in1.Rect();
        rDst &= in2.Rect();
        rDst &= in3.Rect();
        #pragma omp parallel for

        for(int y = rDst.top; y < rDst.bottom; y++)
        {
            TDst* pD = dst.Ptr(y);
            const TIn1* p1 = in1.Ptr(y);
            const TIn2* p2 = in2.Ptr(y);
            const TIn3* p3 = in3.Ptr(y);

            for(int x = rDst.left; x < rDst.right; x++)
                pD[x] = op(p1[x], p2[x], p3[x]);
        }
    }

    template <typename TDst, typename TIn1, typename TIn2, typename TIn3, typename TIn4, typename TOp>
    inline void QuaternaryPointwiseOperator(
        ImageT<TDst>& dst,
        const ImageT<TIn1>& in1,
        const ImageT<TIn2>& in2,
        const ImageT<TIn3>& in3,
        const ImageT<TIn4>& in4,
        TOp& op
    )
    {
        CRect rDst(dst.Rect());
        rDst &= in1.Rect();
        rDst &= in2.Rect();
        rDst &= in3.Rect();
        rDst &= in4.Rect();
        #pragma omp parallel for

        for(int y = rDst.top; y < rDst.bottom; y++)
        {
            TDst* pD = dst.Ptr(y);
            const TIn1* p1 = in1.Ptr(y);
            const TIn2* p2 = in2.Ptr(y);
            const TIn3* p3 = in3.Ptr(y);
            const TIn4* p4 = in4.Ptr(y);

            for(int x = rDst.left; x < rDst.right; x++)
                pD[x] = op(p1[x], p2[x], p3[x], p4[x]);
        }
    }


}
#endif
