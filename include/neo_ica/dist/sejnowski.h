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

template<class ScalarType>
class sejnowski{

private:
    //Fallback
    void compute_means_logp_fb(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType * means_logp) const;
    void compute_phi_fb(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType* phi) const;
    void compute_dphi_fb(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType* dphi) const;
    //SSE3
    void compute_means_logp_sse3(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType * means_logp) const;
    void compute_phi_sse3(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType* phi) const;
    void compute_dphi_sse3(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType* dphi) const;

public:
    sejnowski(int64_t NC, int64_t NF) : NC_(NC), NF_(NF){}
    void compute_means_logp(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType * means_logp) const;
    void compute_phi(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType* phi) const;
    void compute_dphi(int64_t offset, int64_t sample_size, ScalarType * z1, int const * signs, ScalarType* dphi) const;

private:
    int64_t NC_;
    int64_t NF_;
};

}
}


#endif
