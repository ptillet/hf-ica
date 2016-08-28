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

//exp
template<class T>
inline T exp(T x)
{ return std::exp(x); }

template<>
inline __m128 exp(__m128 x)
{ return fmath::exp_ps(x); }

//log
template<class T>
inline T log(T x)
{ return std::log(x); }

template<>
inline __m128 log(__m128 x)
{ return fmath::log_ps(x); }

//tanh
template<class T>
inline T tanh(T x)
{ return std::tanh(x); }

template<>
inline __m128 tanh(__m128 x){
    // -1 + 2 / (1 + exp (-2x));
    __m128 m2x = _mm_mul_ps(_m2, x);
     // z = 2 / (1 + exp(-2x))
    __m128 z = _mm_div_ps(_2, _mm_add_ps(_1, exp(m2x)));
    return _mm_sub_ps(z, _1);
}

//log(1 + e^x)
template<class T>
inline T log_1pe(T x)
{ return (x>0)*x + log(1 + exp(-std::abs(x))); }

template<>
inline __m128 log_1pe(__m128 x)
{
    __m128 mask = _mm_cmpge_ps(x, _m0);
    __m128 xifpos = _mm_and_ps(mask, x);
    __m128 xneg = _mm_or_ps(_m0, x);
    //xifpos + log(_1 + exp(xneg));
    return xifpos + log(_1 + exp(xneg));
}

//sigmoid
template<class T>
inline T sigmoid(T x)
{ T e = exp(-std::abs(x));
  return ((x<0)?e:1)/(1+e); }

template<>
inline __m128 sigmoid(__m128 x)
{
  __m128 mask = _mm_cmpge_ps(x, _m0);
  __m128 xneg = _mm_or_ps(_m0, x);
  __m128 e = exp(xneg);
  __m128 num = _mm_and_ps(mask, _1);
  num = _mm_or_ps(num, _mm_andnot_ps(mask, e));
  return _mm_div_ps(num, _mm_add_ps(_1, e));
}

}

}

#endif
