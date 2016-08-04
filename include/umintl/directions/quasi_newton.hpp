/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_
#define UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <vector>
#include <cmath>


#include "umintl/tools/shared_ptr.hpp"
#include "umintl/optimization_context.hpp"

#include "forwards.h"



namespace umintl{

template<class BackendType>
struct quasi_newton : public direction<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::MatrixType MatrixType;

    virtual std::string info() const{
        return "Quasi-Newton";
    }

    virtual void init(optimization_context<BackendType> & c)
    {
        reinitialize_ = true;

        N_ = c.N();
        Hy_ = BackendType::create_vector(N_);
        s_ = BackendType::create_vector(N_);
        y_ = BackendType::create_vector(N_);
        H_ = BackendType::create_matrix(N_, N_);

        BackendType::set_to_value(Hy_,0,N_);
        BackendType::set_to_value(s_,0,N_);
        BackendType::set_to_value(y_,0,N_);
    }

    virtual void clean(optimization_context<BackendType> &)
    {
        BackendType::delete_if_dynamically_allocated(Hy_);
        BackendType::delete_if_dynamically_allocated(s_);
        BackendType::delete_if_dynamically_allocated(y_);

        BackendType::delete_if_dynamically_allocated(H_);
    }

    void operator()(optimization_context<BackendType> & c){
      //s = x - xm1;
      BackendType::copy(N_,c.x(),s_);
      BackendType::axpy(N_,-1,c.xm1(),s_);

      //y = g - gm1;
      BackendType::copy(N_,c.g(),y_);
      BackendType::axpy(N_,-1,c.gm1(),y_);

      ScalarType ys = BackendType::dot(N_,s_,y_);

      if(reinitialize_)
        BackendType::set_to_diagonal(N_,H_,1);

      ScalarType gamma = 1;

      {
          BackendType::symv(N_,1,H_,y_,0,Hy_);
          ScalarType yHy = BackendType::dot(N_,y_,Hy_);
          ScalarType sg = BackendType::dot(N_,s_,c.gm1());
          ScalarType gHy = BackendType::dot(N_,c.gm1(),Hy_);
          if(ys/yHy>1)
            gamma = ys/yHy;
          else if(sg/gHy<1)
             gamma = sg/gHy;
          else
              gamma = 1;
      }

      BackendType::scale(N_,N_,gamma,H_);
      BackendType::symv(N_,1,H_,y_,0,Hy_);
      ScalarType yHy = BackendType::dot(N_,y_,Hy_);

      //quasi_newton UPDATE
      //H_ += alpha*(s_*Hy' + Hy*s_') + beta*s_*s_';
      ScalarType alpha = -1/ys;
      ScalarType beta = 1/ys + yHy/pow(ys,2);
      BackendType::syr2(N_,alpha,s_,Hy_,H_);
      BackendType::syr1(N_,beta,s_,H_);

      //p = -H_*g
      BackendType::symv(N_,-1,H_,c.g(),0,c.p());

      if(reinitialize_)
          reinitialize_=false;
    }

private:

    size_t N_;

    VectorType Hy_;
    VectorType s_;
    VectorType y_;

    MatrixType H_;

    bool reinitialize_;

};


}

#endif
