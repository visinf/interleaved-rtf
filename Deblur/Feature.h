#ifndef _H_FEATURE_H_
#define _H_FEATURE_H_

#include <random>

#include "Dataset.h"

namespace Deblurring
{
    class Feature
    {
    public:
        // INTERFACE: the type of a pre-processed image.
		typedef ImageRefC<Dataset::InputLabel> PreProcessType;

        // Linear basis function support
        static const size_t UnaryBasisSize    = 1 + Dataset::IDX_BLUR_KERNEL; // every index up to and excluding the blur kernel
        static const size_t PairwiseBasisSize = 1 + 2*Dataset::IDX_BLUR_KERNEL;

        // Explicit threshold testing
        static const size_t NumThresholdTests = 128;

		// Types of feature checks used to compute responses
		static const int UnaryType    = 0;
		static const int PairwiseType = 1;

        Feature(int type_, int channel_, int offx1_, int offy1_,
                int offx2_, int offy2_, double threshold_ = 0.0)
            : type(type_), channel(channel_),
              offx1(offx1_), offy1(offy1_), offx2(offx2_), offy2(offy2_),
              threshold(threshold_)
        {
        }

        Feature() : type(0), channel(0), offx1(0), offy1(0),
            offx2(0), offy2(0), threshold(0.0) {}

        // INTERFACE: Create a new feature instance with the threshold set from the provided value
        Feature WithThreshold(double threshold_) const
        {
            return Feature(type, channel, offx1, offy1, offx2, offy2, threshold_);
        }

        // INTERFACE: Decide whether to branch left or right.
        bool operator()(int x, int y, const PreProcessType& image,
                        const VecCRef<Vector2D<int>>& offsets) const
        {
            return Response(x, y, image, offsets) < threshold;
        }

        double Response(int x, int y, const PreProcessType& image,
                       const VecCRef<Vector2D<int>>& offsets) const
        {
			switch( type )
			{
			case UnaryType:
				{
					return EvaluatePixelValue(x+offsets[0].x+offx1, y+offsets[0].y+offy1, channel, image);
				}
			case PairwiseType:
				{
					const auto pval1 = EvaluatePixelValue(x+offsets[0].x+offx1, y+offsets[0].y+offy1, channel, image);
					const auto pval2 = (offsets.size() < 2) ?
						                        EvaluatePixelValue(x+offx2, y+offy2, channel, image) :
							                    EvaluatePixelValue(x+offsets[1].x+offx2, y+offsets[1].y+offy2, channel, image);
					return (pval1 - pval2);
				}

			default:
				assert(0);
				return 0;
			}
        }

        static double EvaluatePixelValue(int x, int y, int c,
                                         const PreProcessType& prep)
        {
            // clamp
            x = std::min(std::max(0, x), prep.Width()-1);
            y = std::min(std::max(0, y), prep.Height()-1);

            return prep(x,y)[c];
        }

		// INTERFACE: Fill in the linear basis
        static void ComputeBasis(int x, int y, const PreProcessType& prep,
                                 const VecCRef<Vector2D<int>>& offsets, Dataset::FPType* basis)
        {
			// 1) Constant element
			basis[0] = 1.0;

			// 2) Unary basis
			std::copy(&(prep(x,y)[0]), &(prep(x,y)[0]) + Dataset::IDX_BLUR_KERNEL, &basis[1]);

			//for( int i = 0; i < UnaryBasisSize; ++i )
			//	std::cerr << basis[i] << " ";
			//std::cerr << std::endl;

			// 3) Second part of pairwise basis (if applicable)
			if( offsets.size() > 1 ) {
				std::copy(&(prep(x + offsets[1].x, y + offsets[1].y)[0]), 
					&(prep(x + offsets[1].x, y + offsets[1].y)[0]) + Dataset::IDX_BLUR_KERNEL, &basis[1 + Dataset::IDX_BLUR_KERNEL]);
			}
        }

        // INTERFACE: returns a prep-processed input image of type PreProcessType
        static PreProcessType PreProcess(const ImageRefC<Dataset::InputLabel>& input)
        {
            return input;
        }

		static double QuadraticBasis(const PreProcessType& prep, size_t x, size_t y, size_t basisIndex)
		{
			return 1.0;
		}

		static double ComputeQuadraticBasis(const PreProcessType& prep, const Vector2D<int>& i, const Vector2D<int>& j, size_t basisIndex)
		{
			return 1.0;
		}

		static double ComputeQuadraticBasis(const PreProcessType& prep, const Vector2D<int>& i, int basisIndex)
		{
			return 1.0;
		}

        friend std::ostream& operator<<(std::ostream& os, const Feature& feat);
        friend std::istream& operator>>(std::istream& is, Feature& feat);

    private:
        int type, channel;
        int offx1, offy1, offx2, offy2;
        double threshold;
    };

    // SERIALIZATION INTERFACE: writes a feature instance to stream
    inline std::ostream & operator<<(std::ostream& os, const Feature& feat)
    {
        os << feat.type << " " << feat.channel << " "
           << feat.offx1 << " " << feat.offy1 << " "
           << feat.offx2 << " " << feat.offy2 << " "
           << feat.threshold << std::endl;
        return os;
    }

    // SERIALIZATION INTERFACE: reads a feature instance from a stream
    inline std::istream& operator>>(std::istream& is, Feature& feat)
    {
        is >> feat.type >> feat.channel >> feat.offx1 >> feat.offy1
           >> feat.offx2 >> feat.offy2 >> feat.threshold;
        return is;
    }


    // Repeatedly invoked to create feature instances that are then used for branching
    class FeatureSampler
    {
    private:
        std::mt19937                                         mt;
        std::normal_distribution<double>                     dthreshold;
        std::uniform_int_distribution<int>                  doffset;
		std::uniform_int_distribution<int>                  dtype;
        std::uniform_int_distribution<int>                  dchannel_index;
        std::bernoulli_distribution							btest;

    public:
        FeatureSampler()
			: dthreshold(0.0f,1.0f), doffset(-10, +10), dtype(0, 100),
              dchannel_index(0, Dataset::IDX_BLUR_KERNEL-1), btest(0.5)
        {
        }

        // INTERFACE: The type of features
        typedef Feature TFeature;

        // INTERFACE: Instantiates a new randomly drawn feature
        TFeature operator()(int)
        {
			int type = dtype(mt);

			int channel_index = dchannel_index(mt);
			bool centered_test1 = btest(mt);
			bool centered_test2 = btest(mt);

			if( type < 47 ) {
				return Feature(Feature::UnaryType, channel_index, 
					centered_test1 ? 0 : doffset(mt),
					centered_test1 ? 0 : doffset(mt), 
					0, 
					0);
			} else if ( type < 94 ) {
				return Feature(Feature::PairwiseType, channel_index,
					centered_test1 ? 0 : doffset(mt), 
					centered_test1 ? 0 : doffset(mt),
					centered_test2 ? 0 : doffset(mt),
					centered_test2 ? 0 : doffset(mt));
			} else {
				return Feature(Feature::UnaryType, 0, 0, 0, 0, 0);
			}
        }
    };
}

#endif // _H_FEATURE_H_
