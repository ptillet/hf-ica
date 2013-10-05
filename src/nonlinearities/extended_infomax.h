#ifndef NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_
#define NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_

#include <cstddef>

#include "src/fastapprox.h"

namespace curveica{

template<class ScalarType>
struct extended_infomax_ica{
    extended_infomax_ica(std::size_t NC, std::size_t NF) : NC_(NC), NF_(NF){}
    void operator()( ScalarType * z1, ScalarType * b, int const * signs, ScalarType* phi, ScalarType* means_logp) const;
private:
    std::size_t NC_;
    std::size_t NF_;
};

}


#endif
