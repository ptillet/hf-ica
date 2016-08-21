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

namespace neo_ica{
namespace dist{

template<class T>
class sejnowski{

private:
    //Fallback
    void compute_means_logp_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T * means_logp) const;
    void compute_phi_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void compute_dphi_fb(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;
    //SSE3
    void compute_means_logp_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T * means_logp) const;
    void compute_phi_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void compute_dphi_sse3(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;

public:
    sejnowski(int64_t NC, int64_t NF) : NC_(NC), NF_(NF){}
    void compute_means_logp(int64_t offset, int64_t sample_size, T * z1, T* signs, T * means_logp) const;
    void compute_phi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* phi) const;
    void compute_dphi(int64_t offset, int64_t sample_size, T * z1, T* signs, T* dphi) const;

private:
    int64_t NC_;
    int64_t NF_;
};

}
}


#endif
