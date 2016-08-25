#ifndef NEO_ICA_TOOLS_SIMD_HPP_
#define NEO_ICA_TOOLS_SIMD_HPP_

#include <cstddef>
#include <immintrin.h>

namespace neo_ica
{
namespace tools
{

template<class T>
__m128 load_cast_f32(T* ptr);

template<>
__m128 load_cast_f32<float>(float* ptr)
{ return _mm_loadu_ps(ptr); }

template<>
__m128 load_cast_f32<double>(double* ptr)
{
    __m128d xlo = _mm_loadu_pd(ptr);
    __m128d xhi = _mm_loadu_pd(ptr + 2);
    __m128 x = _mm_movelh_ps(_mm_cvtpd_ps(xlo), _mm_cvtpd_ps(xhi));
    return x;
}

template<class T>
void cast_f32_store(T* ptr, __m128 x);

template<>
void cast_f32_store<float>(float* ptr, __m128 x)
{ _mm_storeu_ps(ptr,x); }

template<>
void cast_f32_store<double>(double* ptr, __m128 x)
{
    _mm_storeu_pd(ptr,_mm_cvtps_pd(x));
    _mm_storeu_pd(ptr+2,_mm_cvtps_pd(_mm_movehl_ps(x,x)));
}

}
}

#endif
