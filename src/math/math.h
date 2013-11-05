#ifndef CURVEICA_MATH_H
#define CURVEICA_MATH_H

#include <pmmintrin.h>
//#include "fastapprox.h"
//#include "sse_mathfun.h"
#include "fmath.hpp"

namespace curveica{
	
    namespace math{

        inline __m128 vexp(__m128 x){
            return fmath::exp_ps(x);
        }

        inline __m128 vlog(__m128 x){
            return fmath::log_ps(x);
        }

        inline __m128 vcosh(__m128 x){
            const __m128  c_0_5 = _mm_set1_ps(0.5f);
            return c_0_5 * (vexp(x) + vexp(-x));
        }

        inline __m128 vtanh(__m128 x){
            const __m128  c_1 = _mm_set1_ps(1.0f);
            const __m128  c_2 = _mm_set1_ps(2.0f);
            return -c_1 + c_2 / (c_1 + vexp (-c_2 * x));
        }

    }

}

#endif
