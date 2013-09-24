#ifndef CLICA_H_
#define CLICA_H_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "fmincl/optimization_options.hpp"

namespace parica{

fmincl::optimization_options make_default_options();

template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);

}

#endif
