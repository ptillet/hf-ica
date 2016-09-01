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
#include "neo_ica/dist.h"
#include "neo_ica/tools/round.hpp"
#include "neo_ica/tools/simd.hpp"

namespace neo_ica{

using namespace math;
using namespace tools;


/*
 * ---------------------------
 * Infomax ICA
 * ---------------------------
 */

//template<class T>
//T infomax<T>::logp(T z, T)
//{ return 2*log_1pe(z) - z;}

//template<class T>
//T infomax<T>::phi(T z, T)
//{ T y = sigmoid(z);
//  return 2*y - 1; }

//template<class T>
//T infomax<T>::dphi(T z, T)
//{
//    T y = sigmoid(z);
//    return 2*y*(1 - y);
//}

//template<class T>
//__m128 infomax<T>::logp(__m128 const & z, __m128 const &)
//{   return 2*log_1pe(z) - z; }

//template<class T>
//__m128 infomax<T>::phi(__m128 const &  z, __m128 const &)
//{  return _2*sigmoid(z) - 1; }

//template<class T>
//__m128 infomax<T>::dphi(__m128 const &  z, __m128 const &)
//{
//    __m128 y = sigmoid(z);
//    return 2*y*(1 - y);
//}

template<class T>
T infomax<T>::logp(T z, T)
{ return std::log(std::cosh(z));}

template<class T>
T infomax<T>::phi(T z, T)
{ return std::tanh(z); }

template<class T>
T infomax<T>::dphi(T z, T)
{
    T y = std::tanh(z);
    return 1 - y*y;
}

template<class T>
__m128 infomax<T>::logp(__m128 const & z, __m128 const &)
{   return  _mm_add_ps(log_1pe(_mm_mul_ps(_m2, z)), _mm_add_ps(_mlog2, z)); }

template<class T>
__m128 infomax<T>::phi(__m128 const &  z, __m128 const &)
{  return tanh(z); }

template<class T>
__m128 infomax<T>::dphi(__m128 const &  z, __m128 const &)
{
    __m128 y = tanh(z);
    return 1 - y*y;
}

/*
 * ---------------------------
 * Extended Infomax ICA
 * ---------------------------
 */

template<class T>
T extended_infomax<T>::logp(T z, T k)
{ return .5*z*z + k*(log_1pe(-2*z) - M_LN2 + z);}

template<class T>
T extended_infomax<T>::phi(T z, T k)
{ return z + k*tanh(z); }

template<class T>
T extended_infomax<T>::dphi(T z, T k)
{
    T y = tanh(z);
    return (1 + k) - k*y*y;
}

template<class T>
__m128 extended_infomax<T>::logp(__m128 const & z, __m128 const &  k)
{
    //t1 = .5z^2
    __m128 t1 = _mm_mul_ps(_0_5, _mm_mul_ps(z, z));
    //t2 = (log(1 + exp(-2*z)) - ln(2) + z)
    __m128 t2 = _mm_add_ps(log_1pe(_mm_mul_ps(_m2, z)), _mm_add_ps(_mlog2, z));
    //res = t1 + k*t2
    return _mm_add_ps(t1, _mm_mul_ps(k, t2));
}

template<class T>
__m128 extended_infomax<T>::phi(__m128 const &  z, __m128 const &  k)
{  return _mm_add_ps(z, _mm_mul_ps(k, tanh(z))); }

template<class T>
__m128 extended_infomax<T>::dphi(__m128 const &  z, __m128 const &  k)
{
    __m128 y = tanh(z);
    //res = (1 + k) - k*y*y;
    return _mm_sub_ps(_mm_add_ps(_1, k),
                      _mm_mul_ps(k, _mm_mul_ps(y, y)));
}


/*
 * ---------------------------
 * Fallback
 * ---------------------------
 */
template<class T, template<class> class F>
void dist<T, F>::phi_fb(int64_t off, int64_t NS, T* pz, T* pk, T* res) const{
    for(int64_t c = 0 ; c < NC_ ; ++c){
        T k = pk[c];
        for(int64_t f = off ; f < off + NS ; ++f)
          res[c*NF_+f] = F<T>::phi(pz[c*NF_+f], k);
    }
}

template<class T, template<class> class F>
void dist<T, F>::dphi_fb(int64_t off, int64_t NS, T * pz, T* pk, T* res) const {
    for(int64_t c = 0 ; c < NC_ ; ++c){
        T k = pk[c];
        for(int64_t f = off ; f < off + NS ; ++f)
            res[c*NF_ + f] = F<T>::dphi(pz[c*NF_ + f], k);
    }
}

template<class T, template<class> class F>
void dist<T, F>::mu_fb(int64_t off, int64_t NS, T * pz, T* pk, T* res) const {
    for(int64_t c = 0 ; c < NC_ ; ++c){
        double sum = 0;
        T k = pk[c];
        for(int64_t f = off ; f < off + NS ; ++f)
          sum += F<T>::logp(pz[c*NF_ + f], k);
        res[c] = -sum/NS;
    }
}

/*
 * ---------------------------
 * SSE3
 * ---------------------------
 */
template<class T, template<class> class F>
void dist<T, F>::phi_sse3(int64_t off, int64_t NS, T* pz, T* pk, T* res) const {
    #pragma omp parallel for
    for(int64_t c = 0 ; c < NC_ ; ++c){
        T k = pk[c];
        __m128 vk = _mm_set1_ps((T)k);
        int64_t f = off;
        for(; f < round_to_previous_multiple(off+NS-3,4)  ; f+=4){
            __m128 z = load_cast_f32<T>(&pz[c*NF_+f]);
            cast_f32_store<T>(&res[c*NF_+f],F<T>::phi(z, vk));
        }
        for(; f < off+NS ; ++f)
          res[c*NF_+f] = F<T>::phi(pz[c*NF_+f], k);
    }
}

template<class T, template<class> class F>
void dist<T, F>::dphi_sse3(int64_t off, int64_t NS, T* pz, T* pk, T* res) const {
    #pragma omp parallel for
    for(int64_t c = 0 ; c < NC_ ; ++c){
        T k = pk[c];
        __m128 vk = _mm_set1_ps(k);
        int64_t f = off;
        for(; f < round_to_previous_multiple(off+NS-3,4)  ; f+=4){
            __m128 z = load_cast_f32<T>(&pz[c*NF_+f]);
            cast_f32_store<T>(&res[c*NF_+f],F<T>::dphi(z, vk));
        }
        for(; f < off+NS ; ++f)
          res[c*NF_+f] = F<T>::dphi(pz[c*NF_ + f], k);
    }
}


template<class T, template<class> class F>
void dist<T, F>::mu_sse3(int64_t off, int64_t NS, T* pz, T* pk, T* res) const {
    #pragma omp parallel for
    for(int64_t c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd((double)0);
        T k = pk[c];
        __m128 vk = _mm_set1_ps(k);
        double sum = 0;
        int64_t f = off;
        for(; f < round_to_previous_multiple(off+NS-3,4)  ; f+=4){
            __m128 z = load_cast_f32<T>(&pz[c*NF_+f]);
            __m128 logp = F<T>::logp(z, vk);
            //sum += logp[0] + logp[1] + logp[2] + logp[3]
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(logp));
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(logp,logp)));
        }
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        for(; f < off+NS; ++f)
          sum += F<T>::logp(pz[c*NF_ + f], k);
        res[c] = -sum/NS;
    }
}


template<class T, template<class> class F>
void dist<T, F>::mu(int64_t off, int64_t NS, T * z1, T* signs, T * mu) const
{
    if(cpu.HW_SSE3)
        mu_sse3(off, NS, z1, signs, mu);
    else
        mu_fb(off, NS, z1, signs, mu);
}

template<class T, template<class> class F>
void dist<T, F>::phi(int64_t off, int64_t NS, T * z1, T* signs, T* phi) const
{
    if(cpu.HW_SSE3)
        phi_sse3(off, NS, z1, signs, phi);
    else
        phi_fb(off, NS, z1, signs, phi);
}

template<class T, template<class> class F>
void dist<T, F>::dphi(int64_t off, int64_t NS, T * z1, T* signs, T* dphi) const
{
    if(cpu.HW_SSE3)
        dphi_sse3(off, NS, z1, signs, dphi);
    else
        dphi_fb(off, NS, z1, signs, dphi);
}

template class dist<float, infomax>;
template class dist<double, infomax>;
template class dist<float, extended_infomax>;
template class dist<double, extended_infomax>;

}
