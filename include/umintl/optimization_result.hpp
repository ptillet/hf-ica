/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_OPTIMIZATION_RESULT_HPP_
#define UMINTL_OPTIMIZATION_RESULT_HPP_

#include <cstddef>

namespace umintl{

  /** @brief Simple structure for the optimization results */
  struct optimization_result{
  private:
  public:
      /** @brief enum for the different possible causes of termintion */
      enum termination_cause_type{
          LINE_SEARCH_FAILED,
          STOPPING_CRITERION,
          MAX_ITERATION_REACHED
      };

      /** @brief the final function value */
      double f;
      /** @brief the final number of iterations */
      std::size_t iteration;
      /** @brief the final number of functions evaluation */
      std::size_t n_functions_eval;
      /** @brief the final number of gradient evaluations */
      std::size_t n_gradient_eval;
      /** @brief the cause of the termination */
      termination_cause_type termination_cause;
  };

}

#endif
