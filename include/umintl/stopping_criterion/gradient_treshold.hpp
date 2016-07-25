/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_STOPPING_CRITERION_GRADIENT_TRESHOLD_HPP_
#define UMINTL_STOPPING_CRITERION_GRADIENT_TRESHOLD_HPP_

#include <cmath>

#include "umintl/optimization_context.hpp"

#include "forwards.h"

namespace umintl{

/** @brief Gradient-based stopping criterion
 *
 *  Stops the optimization procedure when the euclidian norm of the gradient accross two successive iterations  is below a threshold
 */
template<class BackendType>
struct gradient_treshold : public stopping_criterion<BackendType>{
    gradient_treshold(double _tolerance = 1e-5) : tolerance(_tolerance){ }
    double tolerance;

    bool operator()(optimization_context<BackendType> & c){
        return BackendType::nrm2(c.N(),c.g()) < tolerance;
    }
};



}

#endif
