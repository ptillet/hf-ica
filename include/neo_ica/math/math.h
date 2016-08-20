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

static const __m128  _2 = _mm_set1_ps(2.f);
static const __m128  _m2 = _mm_set1_ps(-2.0f);
static const __m128 _1 = _mm_set1_ps(1.f);
static const __m128 _m0 = _mm_set1_ps(-0.f);
static const __m128 _0_5 = _mm_set1_ps(0.5);
static const __m128 _mlog2 = _mm_set1_ps(-0.693147);

template<class T>
inline T log_1pe(T x)
{ return (x>0)*x + log(1 + exp(-abs(x))); }

inline __m128 vexp(__m128 x)
{ return fmath::exp_ps(x); }

inline __m128 vlog(__m128 x)
{ return fmath::log_ps(x); }

inline __m128 vtanh(__m128 x){
    // -1 + 2 / (1 + exp (-2x));
    __m128 m2x = _mm_mul_ps(_m2, x);
     // z = 2 / (1 + exp(-2x))
    __m128 z = _mm_div_ps(_2, _mm_add_ps(_1, vexp(m2x)));
    return _mm_sub_ps(z, _1);
}

template<>
inline __m128 log_1pe(__m128 x)
{
    __m128 mask = _mm_cmpge_ps(x, _m0);
    __m128 xifpos = _mm_and_ps(mask, x);
    __m128 xneg = _mm_or_ps(_m0, x);
    return xifpos + vlog(_1 + vexp(xneg));
}

}

}

#endif
