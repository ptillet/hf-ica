/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_OPTIMIZATION_CONTEXT_HPP
#define UMINTL_OPTIMIZATION_CONTEXT_HPP

#include "umintl/tools/shared_ptr.hpp"
#include "umintl/function_wrapper.hpp"
#include <iostream>

namespace umintl{

    /** @brief The optimization context class
     *
     *  Holds the state of the optimization procedure. Typically passed as function argument, to allow easy
     * access to the usual quantities of interest
     */
    template<class BackendType>
    class optimization_context{
    private:
        optimization_context(optimization_context const & other);
        optimization_context& operator=(optimization_context const & other);
    public:
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::MatrixType MatrixType;

        optimization_context(VectorType const & x0, std::size_t dim, model_base<BackendType> & model, detail::function_wrapper<BackendType> * fun) : fun_(fun), model_(model), iter_(0), dim_(dim){
            x_ = BackendType::create_vector(dim_);
            g_ = BackendType::create_vector(dim_);
            p_ = BackendType::create_vector(dim_);
            xm1_ = BackendType::create_vector(dim_);
            gm1_ = BackendType::create_vector(dim_);

            BackendType::copy(dim_,x0,x_);
        }

        model_base<BackendType> & model(){ return model_; }

        detail::function_wrapper<BackendType> & fun() { return *fun_; }
        unsigned int & iter() { return iter_; }
        unsigned int & N() { return dim_; }
        VectorType & x() { return x_; }
        VectorType & g() { return g_; }
        VectorType & xm1() { return xm1_; }
        VectorType & gm1() { return gm1_; }
        VectorType & p() { return p_; }
        ScalarType & val() { return valk_; }
        ScalarType & valm1() { return valkm1_; }
        ScalarType & dphi_0() { return dphi_0_; }
        ScalarType & alpha() { return alpha_; }

        ~optimization_context(){
            BackendType::delete_if_dynamically_allocated(x_);
            BackendType::delete_if_dynamically_allocated(g_);
            BackendType::delete_if_dynamically_allocated(p_);
            BackendType::delete_if_dynamically_allocated(xm1_);
            BackendType::delete_if_dynamically_allocated(gm1_);
        }

    private:
        tools::shared_ptr< detail::function_wrapper<BackendType> > fun_;
        model_base<BackendType> & model_;

        unsigned int iter_;
        unsigned int dim_;

        VectorType x_;
        VectorType g_;
        VectorType p_;
        VectorType xm1_;
        VectorType gm1_;

        ScalarType valk_;
        ScalarType valkm1_;
        ScalarType dphi_0_;

        ScalarType alpha_;
    };
}
#endif
