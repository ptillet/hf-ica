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
    //.5*(exp(x) + exp(-x))
    __m128  c_0_5 = _mm_set1_ps(0.5f);
    __m128 mx = _mm_xor_ps(x, _mm_set1_ps(-0.f));
    __m128 z = _mm_add_ps(vexp(x), vexp(mx));
    return _mm_mul_ps(c_0_5, z);
}

inline __m128 vtanh(__m128 x){
    // -1 + 2 / (1 + exp (-2x));
    __m128  _1 = _mm_set1_ps(1.0f);
    __m128  _2 = _mm_set1_ps(2.0f);
    __m128  _m2 = _mm_set1_ps(-2.0f);
    __m128 m2x = _mm_mul_ps(_m2, x);
     // z = 2 / (1 + exp(-2x))
    __m128 z = _mm_div_ps(_2, _mm_add_ps(_1, vexp(m2x)));
    return _mm_sub_ps(z, _1);
}

}

}

#endif
