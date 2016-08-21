/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cmath>
#include <cstddef>
#include <omp.h>
#include <immintrin.h>
#include <iostream>

#include "neo_ica/backend/cpu_x86.h"
#include "neo_ica/math/math.h"
#include "neo_ica/dist/sejnowski.h"
#include "neo_ica/tools/round.hpp"


namespace neo_ica{
namespace dist{

using namespace std;
using namespace math;

/*
 * ---------------------------
 * Fallback
 * ---------------------------
 */
template<class T>
void sejnowski<T>::compute_phi_fb(int64_t off, int64_t NS, T* pz, T* pk, T* res) const{
    for(int64_t c = 0 ; c < NC_ ; ++c){
        T k = pk[c];
        for(int64_t f = off ; f < off + NS ; ++f)
          res[c*NF_+f] = pz[c*NF_+f] + k*tanh(pz[c*NF_+f]);
    }
}

template<class T>
void sejnowski<T>::compute_dphi_fb(int64_t off, int64_t NS, T * pz, T* pk, T* res) const {
    for(int64_t c = 0 ; c < NC_ ; ++c){
        int k = pk[c];
        for(int64_t f = off ; f < off + NS ; ++f){
          T y = tanh(pz[c*NF_+f]);
          res[c*NF_+f] = (1 + k) - k*y*y;
        }
    }
}

template<class T>
void sejnowski<T>::compute_means_logp_fb(int64_t off, int64_t NS, T * pz, T* pk, T* res) const {
    for(int64_t c = 0 ; c < NC_ ; ++c){
        double sum = 0;
        T k = pk[c];
        for(int64_t f = off ; f < off + NS ; ++f){
          T z = pz[c*NF_+f];
          sum += .5*z*z + k*(log_1pe(-2*z) - M_LN2 + z);
        }
        res[c] = -sum/NS;
    }
}


template<class T>
__m128 load_cast_f32(T* ptr);

template<>
__m128 load_cast_f32<float>(float* ptr)
{ return _mm_loadu_ps(ptr); }

template<>
__m128 load_cast_f32<double>(double* ptr)
{
    __m128d xlo = _mm_load_pd(ptr);
    __m128d xhi = _mm_load_pd(ptr + 2);
    __m128 x = _mm_movelh_ps(_mm_cvtpd_ps(xlo), _mm_cvtpd_ps(xhi));
    return x;
}

template<class T>
void cast_f32_store(T* ptr, __m128 x);

template<>
void cast_f32_store<float>(float* ptr, __m128 x)
{ _mm_store_ps(ptr,x); }

template<>
void cast_f32_store<double>(double* ptr, __m128 x)
{
    _mm_store_pd(ptr,_mm_cvtps_pd(x));
    _mm_store_pd(ptr+2,_mm_cvtps_pd(_mm_movehl_ps(x,x)));
}
/*
 * ---------------------------
 * SSE3
 * ---------------------------
 */
template<class T>
void sejnowski<T>::compute_phi_sse3(int64_t off, int64_t NS, T* pz, T* pk, T* res) const {
    #pragma omp parallel for
    for(int64_t c = 0 ; c < NC_ ; ++c){
        T k = pk[c];
        __m128 vk = _mm_set1_ps((T)k);
        int64_t f = off;
        for(; f < tools::round_to_next_multiple(off,4); ++f)
          res[c*NF_+f] = pz[c*NF_+f] + k*tanh(pz[c*NF_+f]);
        for(; f < tools::round_to_previous_multiple(off+NS,4)  ; f+=4){
            __m128 z = load_cast_f32<T>(&pz[c*NF_+f]);
            __m128 val = _mm_add_ps(z, _mm_mul_ps(vk, vtanh(z)));
            cast_f32_store<T>(&res[c*NF_+f],val);
        }
        for(; f < off+NS ; ++f)
          res[c*NF_+f] = pz[c*NF_+f] + k*tanh(pz[c*NF_+f]);
    }
}

template<class T>
void sejnowski<T>::compute_dphi_sse3(int64_t off, int64_t NS, T* pz, T* pk, T* res) const {
    #pragma omp parallel for
    for(int64_t c = 0 ; c < NC_ ; ++c){
        T k = pk[c];
        __m128 vk = _mm_set1_ps(k);
        int64_t f = off;
        for(; f < tools::round_to_next_multiple(off,4); ++f){
          float y = tanh(pz[c*NF_+f]);
          res[c*NF_+f] = (1 + k) - k*y*y;
        }
        for(; f < tools::round_to_previous_multiple(off+NS,4)  ; f+=4){
            __m128 z = load_cast_f32<T>(&pz[c*NF_+f]);
            __m128 y = vtanh(z);
            //val = (1 + k) - k*y*y;
            __m128 val = _mm_sub_ps(_mm_add_ps(_1, vk),
                                    _mm_mul_ps(vk, _mm_mul_ps(y, y)));
            cast_f32_store<T>(&res[c*NF_+f],val);
        }
        for(; f < off+NS ; ++f){
          float y = tanh(pz[c*NF_+f]);
          res[c*NF_+f] = (1 + k) - k*y*y;
        }
    }
}

template<class T>
void sejnowski<T>::compute_means_logp_sse3(int64_t off, int64_t NS, T* pz, T* pk, T* res) const {
    #pragma omp parallel for
    for(int64_t c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd((double)0);
        T k = pk[c];
        __m128 vk = _mm_set1_ps(k);
        double sum = 0;
        int64_t f = off;
        for(; f < tools::round_to_next_multiple(off,4); ++f){
          T z = pz[c*NF_+f];
          sum += .5*z*z + k*(log(1 + exp(-2*z)) - 0.693147 - z);
        }
        for(; f < tools::round_to_previous_multiple(off+NS,4)  ; f+=4){
            __m128 z = load_cast_f32<T>(&pz[c*NF_+f]);
            //t1 = .5z^2
            __m128 t1 = _mm_mul_ps(_0_5, _mm_mul_ps(z, z));
            //t2 = (log(1 + exp(-2*z)) - ln(2) + z)
            __m128 t2 = _mm_add_ps(log_1pe(_mm_mul_ps(_m2, z)), _mm_add_ps(_mlog2, z));
            //val = t1 + k*t2
            __m128 val = _mm_add_ps(t1, _mm_mul_ps(vk, t2));
            //sum += val[0] + val[1] + val[2] + val[3]
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(val));
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(val,val)));
        }
        for(; f < off+NS; ++f){
          T z = pz[c*NF_+f];
          sum += .5*z*z + k*(log(1 + exp(-2*z)) - 0.693147 - z);
        }
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        res[c] = -sum/NS;
    }
}


template<class T>
void sejnowski<T>::compute_means_logp(int64_t off, int64_t NS, T * z1, T* signs, T * means_logp) const
{
    if(cpu.HW_SSE3)
        compute_means_logp_sse3(off, NS, z1, signs, means_logp);
    else
        compute_means_logp_fb(off, NS, z1, signs, means_logp);
}

template<class T>
void sejnowski<T>::compute_phi(int64_t off, int64_t NS, T * z1, T* signs, T* phi) const
{
    if(cpu.HW_SSE3)
        compute_phi_sse3(off, NS, z1, signs, phi);
    else
        compute_phi_fb(off, NS, z1, signs, phi);
}

template<class T>
void sejnowski<T>::compute_dphi(int64_t off, int64_t NS, T * z1, T* signs, T* dphi) const
{
    if(cpu.HW_SSE3)
        compute_dphi_sse3(off, NS, z1, signs, dphi);
    else
        compute_dphi_fb(off, NS, z1, signs, dphi);
}


template class sejnowski<float>;
template class sejnowski<double>;


}
}
