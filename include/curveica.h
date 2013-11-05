#ifndef curveica_H_
#define curveica_H_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * curveica - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "umintl/optimization_options.hpp"

namespace curveica{

enum optimization_method_type{
    HESSIAN_FREE,
    SD,
    NCG,
    LBFGS,
    BFGS
};

struct options{
    std::size_t max_iter;
    unsigned int verbosity_level;
    optimization_method_type optimization_method;
};

options make_default_options();

template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t NF, options const & opt, double* W = NULL, double* S = NULL);

}

#endif
