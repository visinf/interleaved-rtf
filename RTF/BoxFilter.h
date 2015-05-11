/********************************************************************

Copyright (c)2005 Microsoft Corporation

Module Name:

    BoxFilter.h

Abstract:



Author:

    Toby Sharp (tsharp) 08-Nov-2007

Notes:

********************************************************************/
#pragma once

#include <memory.h>
#include <omp.h>
#include <algorithm>

namespace BoxFilter
{

#undef min
#undef max

// Fixed point arithmetic allows us to multiply instead of dividing
#define SHIFT	((unsigned int)16)
#define MULT	((unsigned int)1 << SHIFT)
#define HALF	(MULT >> 1)

    template <typename TDst, typename TSrc>
    class intscale
    {
    public:
        intscale(const intscale& rhs) : m_scale(rhs.m_scale) {}
        intscale(double scale) : m_scale((int)(scale * MULT)) {}
        TDst operator()(TSrc src) const
        {
            return (TDst)((m_scale * src + HALF) >> SHIFT);
        }
    private:
        int m_scale;
    };

    template <typename TDst, typename TSrc>
    class floatscale
    {
    public:
        floatscale(const floatscale& rhs) : m_scale(rhs.m_scale) {}
        floatscale(double scale) : m_scale(scale) {}
        TDst operator()(TSrc src) const
        {
            return (TDst)(m_scale * src);
        }
    private:
        double m_scale;
    };

    template <typename T> struct type_traits {};
    template <> struct type_traits<unsigned char>
    {
        typedef unsigned int TSum;
        typedef intscale<unsigned char, TSum> TScale;
        template <typename TSrc> struct TScale2 : intscale<unsigned char, TSrc>
        {
            TScale2(double scale) : intscale(scale) {}
            TScale2(const TScale2& rhs) : intscale(rhs) {}
        };
    };
    template <> struct type_traits<unsigned short>
    {
        typedef unsigned int TSum;
        typedef intscale<unsigned short, TSum> TScale;
        template <typename TSrc> struct TScale2 : intscale<unsigned short, TSrc>
        {
            TScale2(double scale) : intscale(scale) {}
            TScale2(const TScale2& rhs) : intscale(rhs) {}
        };
    };
    template <> struct type_traits<float>
    {
        typedef float TSum;
        typedef floatscale<float, TSum> TScale;
        template <typename TSrc> struct TScale2 : floatscale<float, TSrc>
        {
            TScale2(double scale) : floatscale(scale) {}
            TScale2(const TScale2& rhs) : floatscale(rhs) {}
        };
    };
    template <> struct type_traits<double>
    {
        typedef double TSum;
        typedef floatscale<double, TSum> TScale;
        template <typename TSrc> struct TScale2 : floatscale<double, TSrc>
        {
            TScale2(double scale) : floatscale(scale) {}
            TScale2(const TScale2& rhs) : floatscale(rhs) {}
        };
    };


    template <typename T, typename TSum, typename TSumToDst, unsigned char count>
    void BoxFilter1DHelper(T* pDst, TSum* pSum, const T*& pAdd, const T*& pSub,
                           int cx, int nDstInc, int nAddInc, int nSubInc, TSumToDst& sumtodst)
    {
        for (int x = 0; x < (cx); x++, pDst += (nDstInc), pAdd += nAddInc, pSub += nSubInc)
        {
            for (unsigned char c = 0; c < count; c++)
            {
                pSum[c] += pAdd[c] - pSub[c];
                pDst[c] = sumtodst(pSum[c]);
            }
        }
    }

#define BoxFilter1D_Helper(pDst, uSum, pAdd, pSub, cx, nDstInc, nAddInc, nSubInc, count, scale) \
   for (int x = 0; x < (cx); x++, pDst += (nDstInc), pAdd += (nAddInc), pSub += (nSubInc)) \
   { \
      for (unsigned char c = 0; c < count; c++) \
      { \
         uSum[c] += pAdd[c] - pSub[c]; \
         pDst[c] = scale(uSum[c]); \
      } \
   } \
 
    template <typename T, typename TSumToDst, unsigned char ucBands>
    static void BoxFilter1D_Row(	T* pDst,
                                    const T* pSrc,
                                    const int nDStep,
                                    const int nSStep,
                                    int cx,
                                    unsigned short width,
                                    TSumToDst sumtodst)
    {
        // This is where we make the arbitrary decision about how to position filter windows for even widths.
        // (We choose to position them half a pixel to the right of their true centres.)
        unsigned short radiusl = (width - 1) >> 1;
        unsigned short radiusr = width - 1 - radiusl;

        // Prepare the window by summing the left-hand values
        type_traits<T>::TSum uSum[ucBands];
        for (unsigned char uc = 0; uc < ucBands; uc++)
            uSum[uc] = width * pSrc[uc];

        // Read source values into the window before the window is centred on the first pixel
        // Stop when we are centred at the first pixel (we need to start writing), or
        // when we are about to reach the end of the source data (we need to handle the boundary)
        const T* pAdd = pSrc + nSStep;
        const T* pSub = pSrc;
        int xPos1 = std::min(0, cx - radiusr - 1);
        for (int x = 1 - radiusr; x < xPos1; x++, pAdd += nSStep)
        {
            for (unsigned char uc = 0; uc < ucBands; uc++)
                uSum[uc] += pAdd[uc] - pSub[uc];
        }

        // If we reached the end of the source data, we handle the boundary
        // Here we read in any necessary right-hand values prior to arriving at the first pixel
        // (pAdd now points to the last pixel in this case)
        if (xPos1 < 0)
        {
            for (unsigned char uc = 0; uc < ucBands; uc++)
                uSum[uc] -= xPos1 * (pAdd[uc] - pSub[uc]);
        }

        // Now we have arrived at the first pixel and are ready to write data.

        // The first case is to add in source data and subtract left-boundary data.
        // We can do this until either the left edge hits the data or the right edge hits the end
        // Note that we do not enter this loop for the case radius >= cx.
        int xPos2 = std::min(radiusl + 1, cx - radiusr - 1);
        BoxFilter1D_Helper(pDst, uSum, pAdd, pSub, xPos2, nDStep, nSStep, 0, ucBands, sumtodst);

        // If the window fits inside the data, we now add and subtract from the source data (width + 1 < cx).
        BoxFilter1D_Helper(pDst, uSum, pAdd, pSub, cx - width - 1, nDStep, nSStep, nSStep, ucBands, sumtodst);
        // But if the data fits inside the window, we add and subtract boundary data as we write (cx < width + 1).
        BoxFilter1D_Helper(pDst, uSum, pAdd, pSub, width + 1 - cx, nDStep, 0, 0, ucBands, sumtodst);

        // The final stage adds right-boundary data, subtracts source data and writes the result.
        // We can begin to do this as soon as the right edge hits the end and the left edge hits the data
        int xPos3 = std::max(radiusl + 1, cx - radiusr - 1);
        BoxFilter1D_Helper(pDst, uSum, pAdd, pSub, cx - xPos3, nDStep, 0, nSStep, ucBands, sumtodst);
    }

    template <typename T, unsigned char ucBands>
    static void BoxFilter1D(      T* pDst,
                                  const T* pSrc,
                                  int douterstep,
                                  int souterstep,
                                  int outercount,
                                  int dinnerstep,
                                  int sinnerstep,
                                  int innercount,
                                  unsigned short width)
    {
        if (width == 1)
        {
            #pragma omp parallel for
            for (int y = 0; y < outercount; y++)
            {
                T* pD = pDst + y * douterstep;
                const T* pS = pSrc + y * souterstep;
                for (int x = 0; x < innercount; x++, pD += dinnerstep, pS += sinnerstep)
                {
                    for (unsigned char uc = 0; uc < ucBands; uc++)
                        pD[uc] = pS[uc];
                }
            }
        }
        else if (width > 1)
        {
            typedef type_traits<T>::TScale TSumToDst;
            TSumToDst scale(1.0 / (double)width);

            #pragma omp parallel for
            for (int y = 0; y < outercount; y++)
            {
                BoxFilter1D_Row<T, TSumToDst, ucBands>(	pDst + y * douterstep,
                                                        pSrc + y * souterstep,
                                                        dinnerstep,
                                                        sinnerstep,
                                                        innercount,
                                                        width,
                                                        scale);
            }
        }
    }

    template <typename T, unsigned char ucBands>
    static void BoxFilter2DT(	T* pDst,
                                T* pTmp,
                                const T* pSrc,
                                int dstride,
                                int tstride,
                                int sstride,
                                int cx,
                                int cy,
                                unsigned short xwidth,
                                unsigned short ywidth)
    {
        if (xwidth == 0 || ywidth == 0)
        {
            for (int y = 0; y < cy; y++)
                memset(pDst + y * dstride, 0, dstride);
        }
        else if (xwidth == 1 && ywidth == 1)
        {
            for (int y = 0; y < cy; y++)
                memcpy(pDst + y * dstride, pSrc + y * sstride, std::min(sstride, dstride) * sizeof(T));
        }
        else
        {
            BoxFilter1D<T, ucBands>(pTmp, pSrc, ucBands, sstride, cy, tstride, ucBands, cx, xwidth);
            BoxFilter1D<T, ucBands>(pDst, pTmp, ucBands, tstride, cx, dstride, ucBands, cy, ywidth);
        }
    }

    template <typename T, unsigned char ucBands>
    static void BoxFilter2D(	T* pDst,
                                T* pTmp,
                                const T* pSrc,
                                int dstride,
                                int tstride,
                                int sstride,
                                int cx,
                                int cy,
                                unsigned short xwidth,
                                unsigned short ywidth)
    {
        if (xwidth == 0 || ywidth == 0)
        {
            for (int y = 0; y < cy; y++)
                memset(pDst + y * dstride, 0, dstride);
        }
        else if (xwidth == 1 && ywidth == 1)
        {
            for (int y = 0; y < cy; y++)
                memcpy(pDst + y * dstride, pSrc + y * sstride, std::min(sstride, dstride) * sizeof(T));
        }
        else
        {
            BoxFilter1D<T, ucBands>(pTmp, pSrc, tstride, sstride, cy, ucBands, ucBands, cx, xwidth);
            BoxFilter1D<T, ucBands>(pDst, pTmp, ucBands, ucBands, cx, dstride, tstride, cy, ywidth);
        }
    }

    inline void BoxFilter2DT_RGB24(	unsigned char* pDst,
                                    unsigned char* pTmp,
                                    const unsigned char* pSrc,
                                    int dstride,
                                    int tstride,
                                    int sstride,
                                    int cx,
                                    int cy,
                                    unsigned short xwidth,
                                    unsigned short ywidth)
    {
        BoxFilter2DT<unsigned char, 3>(pDst, pTmp, pSrc, dstride, tstride, sstride, cx, cy, xwidth, ywidth);
    }

    inline void BoxFilter2D_RGB24(	unsigned char* pDst,
                                    unsigned char* pTmp,
                                    const unsigned char* pSrc,
                                    int dstride,
                                    int tstride,
                                    int sstride,
                                    int cx,
                                    int cy,
                                    unsigned short xwidth,
                                    unsigned short ywidth)
    {
        BoxFilter2D<unsigned char, 3>(pDst, pTmp, pSrc, dstride, tstride, sstride, cx, cy, xwidth, ywidth);
    }

    inline void BoxFilter2DT_8U(	unsigned char* pDst,
                                    unsigned char* pTmp,
                                    const unsigned char* pSrc,
                                    int dstride,
                                    int tstride,
                                    int sstride,
                                    int cx,
                                    int cy,
                                    unsigned short xwidth,
                                    unsigned short ywidth)
    {
        BoxFilter2DT<unsigned char, 1>(pDst, pTmp, pSrc, dstride, tstride, sstride, cx, cy, xwidth, ywidth);
    }

    inline void BoxFilter2D_8U(	unsigned char* pDst,
                                unsigned char* pTmp,
                                const unsigned char* pSrc,
                                int dstride,
                                int tstride,
                                int sstride,
                                int cx,
                                int cy,
                                unsigned short xwidth,
                                unsigned short ywidth)
    {
        BoxFilter2D<unsigned char, 1>(pDst, pTmp, pSrc, dstride, tstride, sstride, cx, cy, xwidth, ywidth);
    }

}