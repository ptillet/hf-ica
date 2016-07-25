/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_LINE_SEARCH_FORWARDS_H_
#define UMINTL_LINE_SEARCH_FORWARDS_H_

#include <cmath>

#include "umintl/directions/forwards.h"
#include "umintl/directions/quasi_newton.hpp"

#include "umintl/optimization_context.hpp"


namespace umintl{

  template<class ScalarType>
  inline ScalarType cubicmin(ScalarType a,ScalarType b, ScalarType fa, ScalarType fb, ScalarType dfa, ScalarType dfb, ScalarType xmin, ScalarType xmax){
    ScalarType d1 = dfa + dfb - 3*(fa - fb)/(a-b);
    ScalarType delta = pow(d1,2) - dfa*dfb;
    if(delta<0)
      return (xmin+xmax)/2;
    ScalarType d2 = std::sqrt(delta);
    ScalarType x = b - (b - a)*((dfb + d2 - d1)/(dfb - dfa + 2*d2));
    if(std::isnan(x))
      return (xmin+xmax)/2;
    return std::min(std::max(x,xmin),xmax);
  }

  template<class ScalarType>
  inline ScalarType cubicmin(ScalarType a,ScalarType b, ScalarType fa, ScalarType fb, ScalarType dfa, ScalarType dfb){
    return cubicmin(a,b,fa,fb,dfa,dfb,std::min(a,b), std::max(a,b));
  }


  template<class BackendType>
  struct line_search_result{
    private:
      typedef typename BackendType::VectorType VectorType;
      typedef typename BackendType::ScalarType ScalarType;

      //NonCopyable, we do not want useless temporaries here
      line_search_result(line_search_result const &){ }
      line_search_result & operator=(line_search_result const &){ }
    public:
      line_search_result(std::size_t dim) : has_failed(false), best_x(BackendType::create_vector(dim)), best_g(BackendType::create_vector(dim)){ }
      ~line_search_result() {
          BackendType::delete_if_dynamically_allocated(best_x);
          BackendType::delete_if_dynamically_allocated(best_g);
      }
      bool has_failed;
      ScalarType best_alpha;
      ScalarType best_phi;
      VectorType best_x;
      VectorType best_g;
  };

  template<class BackendType>
  struct line_search{
      typedef typename BackendType::ScalarType ScalarType;
      line_search(unsigned int _max_evals) : max_evals(_max_evals){ }
      virtual ~line_search(){ }
      virtual void init(optimization_context<BackendType> &){ }
      virtual void clean(optimization_context<BackendType> &){ }
      virtual void operator()(line_search_result<BackendType> & res,umintl::direction<BackendType> * direction, optimization_context<BackendType> & context) = 0;
  protected:
      unsigned int max_evals;
  };


}

#endif
