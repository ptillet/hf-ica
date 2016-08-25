/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_
#define NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_

#include <cstddef>
#include <stdint.h>
#include <immintrin.h>

namespace neo_ica{

template<class T>
struct extended_infomax
{
    inline static T logp(T z, T k);
    inline static T phi(T z, T k);
    inline static T dphi(T z, T k);

    inline static __m128 logp(__m128 const &  z, __m128 const &  k);
    inline static __m128 phi(__m128 const &  z, __m128 const &  k);
    inline static __m128 dphi(__m128 const & z, __m128 const &  k);
};

template<class T, template<class> class F>
class dist{

private:
    //Fallback
    void mean_logp_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T * mean_logp) const;
    void phi_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void dphi_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;
    //SSE3
    void mean_logp_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T * mean_logp) const;
    void phi_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void dphi_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;

public:
    dist(int64_t NC, int64_t NF) : NC_(NC), NF_(NF){}
    void mean_logp(int64_t offset, int64_t sample_size, T * z1, T* signs, T * mean_logp) const;
    void phi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void dphi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;

private:
    int64_t NC_;
    int64_t NF_;
};

}


#endif
