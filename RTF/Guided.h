#pragma once

#include "Image.h"
#include "BoxFilter.h"

namespace GuidedFilter
{

    template <typename T>
    void BoxFilter(ImageT<T>& dst, const ImageT<T>& src, unsigned short r)
    {
        ImageT<T> tmp(src.Height(), src.Width());

        // TODO: Compare timings for transpose vs. non-transpose
        BoxFilter::BoxFilter2DT<T, 1>(dst.Ptr(), tmp.Ptr(), src.Ptr(),
                                      dst.Stride(), tmp.Stride(), src.Stride(),
                                      dst.Width(), dst.Height(),
                                      2*r + 1, 2*r + 1);
    }

    template <typename T> struct type_traits {
        typedef T TSqr;
    };
    template <> struct type_traits<unsigned char>
    {
        typedef unsigned short TSqr;
        static const unsigned char TMax = 255;
    };
    template <> struct type_traits<unsigned short>
    {
        static const unsigned short TMax = 65535;
    };
    template <> struct type_traits<float>
    {
        static const float TMax;
        typedef float TSqr;
    };
    /* static */
    __declspec(selectany) const float type_traits<float>::TMax = 1.0f;
    template <> struct type_traits<double>
    {
        static const double TMax;
        typedef double TSqr;
    };
    /* static */
    __declspec(selectany) const double type_traits<double>::TMax = 1.0;

    template <typename TDst, typename TSrc> TDst rescale(TSrc a)
    {
        return a;
    }

    template <> unsigned char rescale<unsigned char, double>(double a)
    {
        return (unsigned char)(255.0 * a + 0.5);
    }

    template <> double rescale<double, unsigned char>(unsigned char a)
    {
        return (double)a / 255.0;
    }

    template <> unsigned char rescale<unsigned char, unsigned short>(unsigned short a)
    {
        int ab = a + 128;
        return (unsigned char)((ab + (ab >> 8)) >> 8);
        //return (a + 128) >> 8;
    }

    template <> unsigned short rescale<unsigned short, unsigned char>(unsigned char a)
    {
        return a * 255;
        //return a << 8;
    }

// TODO: Exploit task parallelism
    template <typename T>
    void Filter(ImageT<T>& q, const ImageT<T>& p, const ImageT<T>& I, unsigned short radius, float epsilon)
    {
        typedef type_traits<T>::TSqr TSqr;
        typedef TSqr TVar;

        int cx = p.Width(), cy = p.Height();
        if (cx != I.Width() || cy != I.Height())
            throw;

        // We do 6 box filters: 4 on type T and 2 on type TVar.
        // And we do 6 pointwise operations
        ImageT<T> abar(cx, cy);
        ImageT<T> bbar(cx, cy);
        {
            ImageT<T> mu(cx, cy);
            BoxFilter(mu, I, radius);

            ImageT<TVar> sigmasqr(cx, cy);
            {
                ImageT<TVar> Isqr(cx, cy);
                Arithmetic::UnaryPointwiseOperator(Isqr, I, [] (T lhs) {
                    return rescale<TVar, TSqr>(lhs * lhs);
                } );
                ImageT<TVar> Isqrbar(cx, cy);
                BoxFilter(Isqrbar, Isqr, radius);
                Arithmetic::BinaryPointwiseOperator(sigmasqr, Isqrbar, mu, [] (TVar Isqrbar, T mu)
                {
                    return rescale<TVar, TSqr>(std::max<TSqr>(0, rescale<TSqr, TVar>(Isqrbar) - mu * mu));
                } );
            }

            {
                ImageT<T> pbar(cx, cy);
                BoxFilter(pbar, p, radius);

                ImageT<T> a(cx, cy);
                ImageT<T> b(cx, cy);
                {
                    ImageT<TVar> Ip(cx, cy);
                    ImageT<TVar> Ipbar(cx, cy);
                    Arithmetic::BinaryPointwiseOperator(Ip, I, p, [] (T lhs, T rhs) {
                        return rescale<TVar, TSqr>(lhs * rhs);
                    } );
                    BoxFilter(Ipbar, Ip, radius);

                    TSqr eps = (TSqr)(epsilon * type_traits<T>::TMax * epsilon * type_traits<T>::TMax);
                    Arithmetic::QuaternaryPointwiseOperator(a, Ipbar, mu, pbar, sigmasqr,
                                                            [=] (TVar ipbar, T mu, T pbar, TVar sigmasqr) -> T
                    {
                        TSqr denom = eps + rescale<TSqr, TVar>(sigmasqr);
                        if (denom == (TSqr)0)
                            return type_traits<T>::TMax;
                        TSqr num = std::max<TSqr>(0, rescale<TSqr, TVar>(ipbar) - mu * pbar);
                        TSqr a = std::min<TSqr>((num * type_traits<T>::TMax) / denom, type_traits<T>::TMax);
                        return (T)a;
                    } );
                }

                Arithmetic::TertiaryPointwiseOperator(b, pbar, a, mu, [] (T pbar, T a, T mu) {
                    return pbar - rescale<T, TSqr>(a * mu);
                } );

                // a is in the range [0, 1], while b is in [Tmin, Tmax].
                BoxFilter(abar, a, radius);
                BoxFilter(bbar, b, radius);
            }
        }
        Arithmetic::TertiaryPointwiseOperator(q, abar, I, bbar, [] (T abar, T I, T bbar) {
            return rescale<T, TSqr>(abar * I) + bbar;
        } );
    }

    /*template <typename T>
    void FilterRGB(ImageT<T>& q, const ImageT<T>& p, const ImageT<T, 3>& I, unsigned short radius, float epsilon)
    {
       typedef type_traits<T>::TSqr TSqr;
       typedef TSqr TVar;

       int cx = p.Width(), cy = p.Height();
       if (cx != I.Width() || cy != I.Height())
          throw;

       // We do 6 box filters: 4 on type T and 2 on type TVar.
       // And we do 6 pointwise operations
       // TODO: Exploit task parallelism
       ImageT<T> abar(cx, cy);
       ImageT<T> bbar(cx, cy);
       {
          ImageT<T, 3> mu(cx, cy);
          BoxFilter(mu, I, radius);

          ImageT<TVar> sigmasqr(cx, cy);
          {
             ImageT<TVar> Isqr(cx, cy);
             Arithmetic::UnaryPointwiseOperator(Isqr, I, [] (T lhs) { return rescale<TVar, TSqr>(lhs * lhs); } );
             ImageT<TVar> Isqrbar(cx, cy);
             BoxFilter(Isqrbar, Isqr, radius);
             Arithmetic::BinaryPointwiseOperator(sigmasqr, Isqrbar, mu, [] (TVar Isqrbar, T mu)
             {
                return rescale<TVar, TSqr>(std::max<TSqr>(0, rescale<TSqr, TVar>(Isqrbar) - mu * mu));
             } );
          }

          {
             ImageT<T> pbar(cx, cy);
             BoxFilter(pbar, p, radius);

             ImageT<T> a(cx, cy);
             ImageT<T> b(cx, cy);
             {
                ImageT<TVar> Ip(cx, cy);
                ImageT<TVar> Ipbar(cx, cy);
                Arithmetic::BinaryPointwiseOperator(Ip, I, p, [] (T lhs, T rhs) { return rescale<TVar, TSqr>(lhs * rhs); } );
                BoxFilter(Ipbar, Ip, radius);

                TSqr eps = (TSqr)(epsilon * type_traits<T>::TMax * epsilon * type_traits<T>::TMax);
                Arithmetic::QuaternaryPointwiseOperator(a, Ipbar, mu, pbar, sigmasqr,
                 [=] (TVar ipbar, T mu, T pbar, TVar sigmasqr) -> T
                {
                   TSqr denom = eps + rescale<TSqr, TVar>(sigmasqr);
                   if (denom == (TSqr)0)
                      return type_traits<T>::TMax;
                   TSqr num = std::max<TSqr>(0, rescale<TSqr, TVar>(ipbar) - mu * pbar);
                   TSqr a = std::min<TSqr>((num * type_traits<T>::TMax) / denom, type_traits<T>::TMax);
                   return (T)a;
                } );
             }

             Arithmetic::TertiaryPointwiseOperator(b, pbar, a, mu, [] (T pbar, T a, T mu) { return pbar - rescale<T, TSqr>(a * mu); } );

             // a is in the range [0, 1], while b is in [Tmin, Tmax].
             BoxFilter(abar, a, radius);
             BoxFilter(bbar, b, radius);
          }
       }
       Arithmetic::TertiaryPointwiseOperator(q, abar, I, bbar, [] (T abar, T I, T bbar) { return rescale<T, TSqr>(abar * I) + bbar; } );
    }*/

}
