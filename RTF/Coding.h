// File:   Coding.h
// Author: t-jejan
//
// Implements routines for mapping continuous values to a discrete alphabet and back.
//
#ifndef _H_CODING_H_
#define _H_CODING_H_

#include <map>
#include <cmath>
#include <vector>

#include "Training.h"

namespace Coding
{
    template <typename TInValue>
    struct AbsoluteDistance
    {
        static TInValue distance(TInValue a, TInValue b)
        {
            return std::abs(a - b);
        }
    };

    template <typename TInValue, typename TOutValue, size_t Digits, typename TDistance = AbsoluteDistance<TInValue> >
    class CanonicalEncoding
    {
    private:
        std::vector<TInValue>      indexToValue;
        std::map<TInValue, size_t> valueToIndex;

        Training::LabelVector<TOutValue, Digits> kthCanonicalLabel(size_t k) const
        {
            Training::LabelVector<TOutValue, Digits> ret;

            for(size_t i = 0; i < Digits; ++i)
                ret[i] = (i == k) ? TOutValue(1) : TOutValue(0);

            return ret;
        }

    public:
        Training::LabelVector<TOutValue, Digits> GetLabel(TInValue value)
        {
            // Check if we already allocated a canonical label for the given value; if so, return it.
            if(valueToIndex.find(value) != valueToIndex.end())
                return kthCanonicalLabel(valueToIndex[value]);

            // Otherwise, if there are still canonical labels available, allocate one for the given value.
            if(indexToValue.size() < Digits)
            {
                indexToValue.push_back(value);
                valueToIndex[value] = indexToValue.size() - 1;
                return kthCanonicalLabel(indexToValue.size() - 1);
            }

            // If no more canonical labels are available, search for the closest value and return the label of that one.
            std::cerr << "Insufficient number of discrete labels!" << std::endl;
            throw std::string("labels");
#if 0
            TInValue mindist = std::numeric_limits<TInValue>::max();
            size_t   closest;

            for(size_t i = 0; i < indexToValue.size(); ++i)
            {
                const auto dist = TDistance::distance(indexToValue[i], value);

                if(dist < mindist)
                {
                    mindist = dist;
                    closest = i;
                }
            }

            return kthCanonicalLabel(closest);
#endif
        }

        TInValue GetValue(const Training::LabelVector<TOutValue, Digits>& label)
        {
            if(indexToValue.size() != Digits)
            {
                std::cerr << "number of canonical labels: " << indexToValue.size();
                throw std::string("too many labels");
            }

            // Find the maximizing index
            TOutValue max = -std::numeric_limits<TOutValue>::max();
            size_t argmax = 0;

            for(size_t k = 0; k < indexToValue.size(); ++k)
            {
                if(label[k] > max)
                {
                    max    = label[k];
                    argmax = k;
                }
            }

            return indexToValue[argmax];
        }
    };
}

#endif // _H_CODING_H_