/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * DSHF-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_
#define NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_

#include <cstddef>

namespace dshf_ica{

template<class ScalarType>
struct extended_infomax_ica{
    extended_infomax_ica(std::size_t NC, std::size_t NF) : NC_(NC), NF_(NF){}
    void compute_means_logp(std::size_t offset, std::size_t sample_size, ScalarType * z1, int const * signs, ScalarType * means_logp) const;
    void compute_phi(std::size_t offset, std::size_t sample_size, ScalarType * z1, int const * signs, ScalarType* phi) const;
    void compute_dphi(std::size_t offset, std::size_t sample_size, ScalarType * z1, int const * signs, ScalarType* dphi) const;
private:
    std::size_t NC_;
    std::size_t NF_;
};

}


#endif
