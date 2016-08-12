/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_
#define UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "umintl/linear/conjugate_gradient.hpp"
#include "umintl/tools/shared_ptr.hpp"
#include "forwards.h"



namespace umintl{

namespace tag{

namespace truncated_newton{

enum stopping_criterion{
    STOP_RESIDUAL_TOLERANCE,
    STOP_HV_VARIANCE
};

}

}


template<class BackendType>
struct truncated_newton : public direction<BackendType>{
  private:
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;

    struct compute_Ab: public linear::conjugate_gradient_detail::compute_Ab<BackendType>{
        compute_Ab(VectorType const & x, VectorType const & g, model_base<BackendType> const & model, umintl::detail::function_wrapper<BackendType> & fun) : x_(x), g_(g), model_(model), fun_(fun){ }
        virtual void operator()(size_t, typename BackendType::VectorType const & b, typename BackendType::VectorType & res){
          fun_.compute_hv_product(x_,g_,b,res,model_.get_hv_product_tag());
        }
      protected:
        VectorType const & x_;
        VectorType const & g_;
        model_base<BackendType> const & model_;
        umintl::detail::function_wrapper<BackendType> & fun_;
    };

    struct variance_stop_criterion : public linear::conjugate_gradient_detail::stopping_criterion<BackendType>{
      private:
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::ScalarType ScalarType;
      public:
        variance_stop_criterion(optimization_context<BackendType> & c) : c_(c){
          psi_=0;
        }

        void init(VectorType const & p0){
          VectorType var = BackendType::create_vector(c_.N());

          size_t H = c_.model().get_hv_product_tag().sample_size;
          size_t offset = c_.model().get_hv_product_tag().offset;
          c_.fun().compute_hv_product_variance(c_.x(),p0,var,hv_product_variance(STOCHASTIC,H,offset));
          ScalarType nrm2p0 = BackendType::nrm2(c_.N(),p0);
          ScalarType nrm1var = BackendType::asum(c_.N(),var);
          gamma_ = nrm1var/(H*std::pow(nrm2p0,2));

          BackendType::delete_if_dynamically_allocated(var);
        }

        void update(VectorType const & dk){
          psi_ = gamma_*std::pow(BackendType::nrm2(c_.N(),dk),2);
        }

        bool operator()(ScalarType rsn){
          return rsn <= psi_;
        }

      private:
        optimization_context<BackendType> & c_;
        ScalarType psi_;
        ScalarType gamma_;
    };

  public:
    truncated_newton(tag::truncated_newton::stopping_criterion _stop = tag::truncated_newton::STOP_RESIDUAL_TOLERANCE, size_t _iter = 0) : iter(_iter), stop(_stop){ }

    virtual std::string info() const{
        return "Truncated Newton";
    }

    void operator()(optimization_context<BackendType> & c){
      if(iter==0) iter = c.N();

      linear::conjugate_gradient<BackendType> solver(iter, new compute_Ab(c.x(), c.g(),c.model(),c.fun()));
      if(stop==tag::truncated_newton::STOP_RESIDUAL_TOLERANCE){
          ScalarType tol = std::min((ScalarType)0.5,std::sqrt(BackendType::nrm2(c.N(),c.g())))*BackendType::nrm2(c.N(),c.g());
          solver.stop = new linear::conjugate_gradient_detail::residual_norm<BackendType>(tol);
      }
      else{
          solver.stop = new variance_stop_criterion(c);
      }

      VectorType minus_g = BackendType::create_vector(c.N());
      BackendType::copy(c.N(),c.g(),minus_g);
      BackendType::scale(c.N(),-1,minus_g);
      BackendType::scale(c.N(),c.alpha(),c.p());


      typename linear::conjugate_gradient<BackendType>::optimization_result res = solver(c.N(),c.p(),minus_g,c.p());
      if(res.i==0 && res.ret == umintl::linear::conjugate_gradient<BackendType>::FAILURE_NON_POSITIVE_DEFINITE)
        BackendType::copy(c.N(),minus_g,c.p());
      //std::cout << res.ret << " " << res.i << std::endl;

      BackendType::delete_if_dynamically_allocated(minus_g);
    }

    size_t iter;
    tag::truncated_newton::stopping_criterion stop;
};

}

#endif
