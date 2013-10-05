#ifndef NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_
#define NONLINEARITIES_EXTENDED_INFOMAX_ICA_H_

#include <cstddef>

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
namespace curveica{

template<class ScalarType>
struct extended_infomax_ica{
    extended_infomax_ica(std::size_t NC, std::size_t NF) : NC_(NC), NF_(NF), phi_(NF_, NC_), z1_(NF_, NC_), b_(NC_), kurt_(NC_), means_logp_(NC_){}
    void operator()( ScalarType * z1, ScalarType * b, int const * signs, ScalarType* phi, ScalarType* means_logp) const;
private:
    std::size_t NC_;
    std::size_t NF_;

    viennacl::matrix<ScalarType, viennacl::column_major> phi_;
    viennacl::matrix<ScalarType, viennacl::column_major> z1_;

    viennacl::vector<ScalarType> b_;
    viennacl::vector<ScalarType> kurt_;
    viennacl::vector<ScalarType> means_logp_;
};

}


#endif
