/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_STOPPING_CRITERION_VALUE_TRESHOLD_HPP_
#define UMINTL_STOPPING_CRITERION_VALUE_TRESHOLD_HPP_

#include <cmath>

#include "umintl/optimization_context.hpp"
#include "forwards.h"

namespace umintl{

/** @brief parameter-based stopping criterion
 *
 *  Stops the optimization procedure when the change in value accross two successive iterations  is below a threshold
 */
template<class BackendType>
struct value_treshold : public stopping_criterion<BackendType>{
    value_treshold(double _tolerance = 1e-5) : tolerance(_tolerance){ }
    double tolerance;

    bool operator()(optimization_context<BackendType> & c){
        return std::fabs(c.val() - c.valm1()) < tolerance;
    }
};

}

#endif
