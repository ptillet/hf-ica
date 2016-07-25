/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DEBUG_HPP
#define UMINTL_DEBUG_HPP

#include "tools/shared_ptr.hpp"
#include "umintl/model_base.hpp"
#include <iostream>

#include <cmath>

namespace umintl{

template<class BackendType, class FUN>
typename BackendType::ScalarType check_grad(FUN & fun, typename BackendType::VectorType const & x0, std::size_t N, typename BackendType::ScalarType h){
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    VectorType x = BackendType::create_vector(N);
    BackendType::copy(N,x0,x);
    VectorType fgrad = BackendType::create_vector(N);
    VectorType dummy = BackendType::create_vector(N);
    VectorType numgrad = BackendType::create_vector(N);
    ScalarType res = 0;
    ScalarType vl, vr;
    umintl::deterministic<BackendType> model;
    fun(x,vl,fgrad,model.get_value_gradient_tag());
    for(unsigned int i=0 ; i < N ; ++i){
        ScalarType vx = x[i];
        x[i] = vx-h; fun(x,vl,dummy,model.get_value_gradient_tag());
        x[i] = vx+h; fun(x,vr,dummy,model.get_value_gradient_tag());
        numgrad[i] = (vr-vl)/(2*h);
        x[i]=vx;
    }
    for(unsigned int i=0 ; i < N ; ++i){
        ScalarType denom = std::max(std::fabs((double)numgrad[i]),std::abs((double)fgrad[i]));
        ScalarType diff = std::fabs(numgrad[i]-fgrad[i]);
        //std::cout << numgrad[i] << " " << fgrad[i] << std::endl;
        if(denom>1)
            diff/=denom;
        res = std::max(res,diff);
    }
    return res;
}

//template<class BackendType, class FUN>
//typename BackendType::ScalarType check_grad_variance(FUN & fun, typename BackendType::VectorType const & x0, std::size_t N){
//    typedef typename BackendType::ScalarType ScalarType;
//    typedef typename BackendType::VectorType VectorType;

//    VectorType var = BackendType::create_vector(N);
//    VectorType tmp = BackendType::create_vector(N);
//    VectorType Hv = BackendType::create_vector(N);
//    BackendType::set_to_value(var,0,N);
//    c.fun().compute_hv_product(x0,c.g(),c.g(),Hv,hessian_vector_product(STOCHASTIC,S,offset_));

//    for(std::size_t i = 0 ; i < S ; ++i){
//        //tmp = (grad(xi) - grad(X)).^2
//        //var += tmp
//        c.fun().compute_hv_product(x0,c.g(),c.g(),tmp,hessian_vector_product(STOCHASTIC,1,offset_+i));
//        for(std::size_t i = 0 ; i < N ; ++i)
//            var[i]+=std::pow(tmp[i]-Hv[i],2);
//    }
//    BackendType::scale(N,(ScalarType)1/(S-1),var);
//    for(std::size_t i = 0 ; i < N ; ++i)
//        std::cout << var[i] << " ";
//    std::cout << std::endl;

//    c.fun().compute_hv_product_variance(x0,c.g(), var, hv_product_variance(STOCHASTIC,S,offset_));

//    for(std::size_t i = 0 ; i < N ; ++i)
//        std::cout << var[i] << " ";
//    std::cout << std::endl;

//    BackendType::delete_if_dynamically_allocated(var);
//    BackendType::delete_if_dynamically_allocated(tmp);
//    BackendType::delete_if_dynamically_allocated(Hv);
//}

}

#endif
