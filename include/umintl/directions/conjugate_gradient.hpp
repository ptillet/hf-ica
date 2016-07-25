/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_
#define UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_

#include "umintl/optimization_context.hpp"

#include "umintl/tools/shared_ptr.hpp"
#include "umintl/directions/forwards.h"


namespace umintl{

namespace tag{

namespace conjugate_gradient{

enum restart{
    NO_RESTART,
    RESTART_ON_DIM,
    RESTART_NOT_ORTHOGONAL
};

enum update{
    UPDATE_POLAK_RIBIERE,
    UPDATE_GILBERT_NOCEDAL,
    UPDATE_FLETCHER_REEVES
};

}

}

template<class BackendType>
struct conjugate_gradient : public direction<BackendType>{
public:
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;
private:
    ScalarType update_polak_ribiere(optimization_context<BackendType> & c){
        VectorType tmp = BackendType::create_vector(c.N());
        BackendType::copy(c.N(),c.g(), tmp);
        BackendType::axpy(c.N(),-1,c.gm1(),tmp);
        ScalarType res = std::max(BackendType::dot(c.N(),c.g(),tmp)/BackendType::dot(c.N(),c.gm1(),c.gm1()),(ScalarType)0);
        BackendType::delete_if_dynamically_allocated(tmp);
        return res;
    }

    ScalarType update_fletcher_reeves(optimization_context<BackendType> & c){
        return BackendType::dot(c.N(),c.g(),c.g())/BackendType::dot(c.N(),c.gm1(),c.gm1());
    }

    ScalarType update_impl(optimization_context<BackendType> & c){
        switch (update) {
            case tag::conjugate_gradient::UPDATE_POLAK_RIBIERE: return update_polak_ribiere(c);
            case tag::conjugate_gradient::UPDATE_GILBERT_NOCEDAL: return std::min(update_polak_ribiere(c), update_fletcher_reeves(c));
            case tag::conjugate_gradient::UPDATE_FLETCHER_REEVES: return update_fletcher_reeves(c);
            default: throw exceptions::incompatible_parameters("Unsupported conjugate gradient update");
        }
    }

    ScalarType restart_on_dim(optimization_context<BackendType> & c){
        return c.iter()==c.N();
    }

    ScalarType restart_not_orthogonal(optimization_context<BackendType> & c){
        double threshold = 0.1;
        return std::abs(BackendType::dot(c.N(),c.g(),c.gm1()))/BackendType::dot(c.N(),c.g(),c.g()) > threshold;
    }

    ScalarType restart_impl(optimization_context<BackendType> & c){
        switch (restart) {
            case tag::conjugate_gradient::NO_RESTART: return false;
            case tag::conjugate_gradient::RESTART_ON_DIM: return restart_on_dim(c);
            case tag::conjugate_gradient::RESTART_NOT_ORTHOGONAL: return restart_not_orthogonal(c);
            default: throw exceptions::incompatible_parameters("Unsupported conjugate gradient restart");
        }
    }



public:
    conjugate_gradient(tag::conjugate_gradient::update _update = tag::conjugate_gradient::UPDATE_POLAK_RIBIERE
            , tag::conjugate_gradient::restart _restart = tag::conjugate_gradient::RESTART_NOT_ORTHOGONAL) : update(_update), restart(_restart){ }

    virtual std::string info() const{
        return "Nonlinear Conjugate Gradient";
    }

    void operator()(optimization_context<BackendType> & c){
        ScalarType beta;
        if(restart_impl(c))
            beta = 0;
        else
            beta = update_impl(c);
        BackendType::scale(c.N(),beta,c.p());
        BackendType::axpy(c.N(),-1,c.g(),c.p());
    }

    tag::conjugate_gradient::update update;
    tag::conjugate_gradient::restart restart;
};

}

#endif
