/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef NEOICA_MATH_H
#define NEOICA_MATH_H

#include <pmmintrin.h>
//#include "fastapprox.h"
//#include "sse_mathfun.h"
#include "fmath.hpp"

namespace neo_ica
{
	
namespace math
{

inline __m128 vexp(__m128 x)
{ return fmath::exp_ps(x); }

inline __m128 vlog(__m128 x)
{ return fmath::log_ps(x); }

inline __m128 vcosh(__m128 x){
    const __m128  c_0_5 = _mm_set1_ps(0.5f);
    return c_0_5 * (vexp(x) + vexp(-x));
}

inline __m128 vtanh(__m128 x){
    const __m128  _1 = _mm_set1_ps(1.0f);
    const __m128  _2 = _mm_set1_ps(2.0f);
    return -_1 + _2 / (_1 + vexp (-_2 * x));
}

}

}

#endif
