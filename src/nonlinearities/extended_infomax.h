#ifndef NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_
#define NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_

#include <cstddef>

namespace curveica{

template<class ScalarType>
struct extended_infomax_ica{
    extended_infomax_ica(std::size_t NC, std::size_t NF) : NC_(NC), NF_(NF){}
    void compute_means_logp(ScalarType * z1, int const * signs, ScalarType * means_logp) const;
    void compute_phi( ScalarType * z1, int const * signs, ScalarType* phi) const;
    void compute_dphi( ScalarType * z1, int const * signs, ScalarType* dphi) const;
private:
    std::size_t NC_;
    std::size_t NF_;
};

}


#endif
