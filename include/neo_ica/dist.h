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

#define DECLARE_NONLINEARITY(NAME) \
    template<class T>\
    struct NAME\
    {\
        inline static T logp(T z, T k);\
        inline static T phi(T z, T k);\
        inline static T dphi(T z, T k);\
\
        inline static __m128 logp(__m128 const &  z, __m128 const &  k);\
        inline static __m128 phi(__m128 const &  z, __m128 const &  k);\
        inline static __m128 dphi(__m128 const & z, __m128 const &  k);\
    }

DECLARE_NONLINEARITY(infomax);
DECLARE_NONLINEARITY(extended_infomax);

template<class T>
class dist_base{
public:
    dist_base(int64_t NC, int64_t NF) : NC_(NC), NF_(NF){}
    virtual void mu(int64_t offset, int64_t sample_size, T * z1, T* signs, T * mu) const = 0;
    virtual void phi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const = 0;
    virtual void dphi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const = 0;

protected:
    int64_t NC_;
    int64_t NF_;
};

template<class T, template<class> class F>
class dist: public dist_base<T>{
    using dist_base<T>::NC_;
    using dist_base<T>::NF_;

private:
    //Fallback
    void mu_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T * mu) const;
    void phi_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void dphi_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;
    //SSE3
    void mu_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T * mu) const;
    void phi_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void dphi_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;

public:
    dist(int64_t NC, int64_t NF) : dist_base<T>(NC, NF){}
    void mu(int64_t offset, int64_t sample_size, T * z1, T* signs, T * mu) const;
    void phi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void dphi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;
};

}


#endif
