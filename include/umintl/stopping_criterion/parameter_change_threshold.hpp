/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_STOPPING_CRITERION_PARAMETER_CHANGE_TRESHOLD_HPP_
#define UMINTL_STOPPING_CRITERION_PARAMETER_CHANGE_TRESHOLD_HPP_

#include <cmath>

#include "umintl/optimization_context.hpp"
#include "forwards.h"

namespace umintl{

/** @brief parameter-based stopping criterion
 *
 *  Stops the optimization procedure when the euclidian norm of the change of parameters accross two successive iterations is below a threshold
 */
template<class BackendType>
struct parameter_change_threshold : public stopping_criterion<BackendType>{
    parameter_change_threshold(double _tolerance = 1e-5) : tolerance(_tolerance){ }
    double tolerance;
    bool operator()(optimization_context<BackendType> & c){
        typename BackendType::VectorType tmp = BackendType::create_vector(c.N());
        BackendType::copy(c.N(),c.x(),tmp);
        BackendType::axpy(c.N(),-1,c.xm1(),tmp);
        double change = BackendType::nrm2(c.N(),tmp);
        BackendType::delete_if_dynamically_allocated(tmp);
        return  change < tolerance;
    }
};

}

#endif
