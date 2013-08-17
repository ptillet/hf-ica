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

#define FMINCL_WITH_EIGEN
#include "fmincl/optimization_otions.hpp"

namespace clica{


fmincl::optimization_options make_default_options();

template<class T, class U>
void whiten(T & data, U & out);

template<class T, class U>
void inplace_linear_ica(T & data, U & out, fmincl::optimization_options const & options = make_default_options());

}

#endif
