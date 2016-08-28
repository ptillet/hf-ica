/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_LINE_SEARCH_STRONG_WOLFE_POWELL_HPP_
#define UMINTL_LINE_SEARCH_STRONG_WOLFE_POWELL_HPP_

#include "umintl/directions/conjugate_gradient.hpp"
#include "umintl/directions/steepest_descent.hpp"
#include "umintl/directions/quasi_newton.hpp"
#include "umintl/directions/truncated_newton.hpp"


#include "umintl/optimization_context.hpp"
#include "forwards.h"

#include <cmath>

#include <map>

namespace umintl{

/** @brief The strong wolfe-powell line-search class
 *  @tparam BackendType the linear algebra backend of the minimizer
 */
template<class BackendType>
struct strong_wolfe_powell : public line_search<BackendType>{
    //Tag
    /** @brief The constructor
     *  @param _max_evals maximum number of value-gradient evaluation in the line-search
     */
    strong_wolfe_powell(unsigned int _max_evals = 40) : line_search<BackendType>(_max_evals) { }

    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::MatrixType MatrixType;

    /** @brief initialization of the temporaries */
    virtual void init(optimization_context<BackendType> & c){
        x0_ = BackendType::create_vector(c.N());
    }

    /** @brief deletion of the temporaries */
    virtual void clean(optimization_context<BackendType> &){
        BackendType::delete_if_dynamically_allocated(x0_);
    }


private:
    using line_search<BackendType>::max_evals;

    /** @brief Sufficient decrease test for the strong wolfe-powell conditions */
    bool sufficient_decrease(ScalarType alpha, ScalarType phi_alpha, ScalarType phi0) const {
        return phi_alpha <= (phi0 + c1_*alpha );
    }

    /** @brief Curvature test for the strong wolfe-powell conditions */
    bool curvature(ScalarType dphi_alpha, ScalarType dphi0) const{
        return std::abs(dphi_alpha) <= c2_*std::abs(dphi0);
    }

    void zoom(line_search_result<BackendType> & res, ScalarType alpha_low, ScalarType phi_alpha_low, ScalarType dphi_alpha_low
              ,ScalarType alpha_high, ScalarType phi_alpha_high, ScalarType dphi_alpha_high
              ,optimization_context<BackendType> & c, unsigned int eval_offset) const{
        VectorType & current_x = res.best_x;
        VectorType & current_g = res.best_g;
        ScalarType & current_phi = res.best_phi;
        VectorType const & p = c.p();
        ScalarType eps = (ScalarType)1e-8;
        ScalarType alpha = 0;
        ScalarType dphi = 0;
        bool twice_close_to_boundary=false;
        for(unsigned int i = eval_offset ; i < max_evals ; ++i){
            ScalarType xmin = std::min(alpha_low,alpha_high);
            ScalarType xmax = std::max(alpha_low,alpha_high);
            if(alpha_low < alpha_high)
                alpha = cubicmin(alpha_low, alpha_high, phi_alpha_low, phi_alpha_high, dphi_alpha_low, dphi_alpha_high,xmin,xmax);
            else
                alpha = cubicmin(alpha_high, alpha_low, phi_alpha_high, phi_alpha_low, dphi_alpha_high, dphi_alpha_low,xmin,xmax);
            if(std::min(xmax - alpha, alpha - xmin)/(xmax - xmin)  < eps){
                res.best_alpha = alpha;
                res.has_failed=true;
                return;
            }
            if(std::min(xmax - alpha, alpha - xmin)/(xmax - xmin) < 0.1){
                if(twice_close_to_boundary){
                    if(std::abs(alpha - xmax) < std::abs(alpha - xmin))
                        alpha = xmax - 0.1*(xmax-xmin);
                    else
                        alpha = xmin + 0.1*(xmax-xmin);
                    twice_close_to_boundary = false;
                }
                else{
                    twice_close_to_boundary = true;
                }
            }
            else{
                twice_close_to_boundary = false;
            }

            //Compute phi(alpha) = f(x0 + alpha*p)
            BackendType::copy(c.N(),x0_,current_x);
            BackendType::axpy(c.N(),alpha,p,current_x);
            c.fun().compute_value_gradient(current_x,current_phi,current_g,c.model().get_value_gradient_tag());
            dphi = BackendType::dot(c.N(),current_g,p);

            if(!sufficient_decrease(alpha,current_phi, c.val()) || current_phi >= phi_alpha_low){
                alpha_high = alpha;
                phi_alpha_high = current_phi;
                dphi_alpha_high = dphi;

            }
            else{
                if(curvature(dphi, c.dphi_0())){
                    res.best_alpha = alpha;
                    res.has_failed = false;
                    return;
                }
                if(dphi*(alpha_high - alpha_low) >= 0){
                    alpha_high = alpha_low;
                    phi_alpha_high = phi_alpha_low;
                    dphi_alpha_high = dphi_alpha_low;
                }
                alpha_low = alpha;
                phi_alpha_low = current_phi;
                dphi_alpha_low = dphi;
            }
        }
        res.best_alpha = alpha;
        res.has_failed=true;
    }

public:

    /** @brief Line-Search procedure call
    *
    * @param res reference to line search result
    * @param direction the descent direction procedure used for the line search
    * @param c corresponding optimization context
    */
    void operator()(line_search_result<BackendType> & res, umintl::direction<BackendType> * direction, optimization_context<BackendType> & c) {
        ScalarType alpha;
        c1_ = (ScalarType)1e-4;
        if(dynamic_cast<conjugate_gradient<BackendType>* >(direction) || dynamic_cast<steepest_descent<BackendType>* >(direction)){
            c2_ = (ScalarType)0.2;
            alpha = std::min((ScalarType)(1.0),1/BackendType::asum(c.N(),c.g()));
        }
        else{
            c2_ = (ScalarType)0.9;
            alpha = 1;
        }

        ScalarType alpham1 = 1e-3;
        ScalarType phi_0 = c.val();
        ScalarType dphi_0 = c.dphi_0();
        ScalarType last_phi = phi_0;
        ScalarType dphim1 = dphi_0;
        ScalarType dphi;


        ScalarType & current_phi = res.best_phi;
        VectorType & current_x = res.best_x;
        VectorType & current_g = res.best_g;
        VectorType const & p = c.p();

        BackendType::copy(c.N(),c.x(), x0_);


        for(unsigned int i = 1 ; i< max_evals; ++i){
            //Compute phi(alpha) = f(x0 + alpha*p) ; dphi = grad(phi)_alpha'*p
            BackendType::copy(c.N(),x0_,current_x);
            BackendType::axpy(c.N(),alpha,p,current_x);
            c.fun().compute_value_gradient(current_x,current_phi,current_g,c.model().get_value_gradient_tag());
            dphi = BackendType::dot(c.N(),current_g,p);

            //Tests sufficient decrease
            if(!sufficient_decrease(alpha, current_phi, phi_0) || (i==1 && current_phi >= last_phi)){
                return zoom(res, alpham1, last_phi, dphim1, alpha, current_phi, dphi, c, i);
            }

            //Tests curvature
            if(curvature(dphi, dphi_0)){
                res.has_failed = false;
                res.best_alpha = alpha;
                return;
            }
            if(dphi>=0){
                return zoom(res, alpha, current_phi, dphi, alpham1, last_phi, dphim1, c, i);
            }

            //Updates context_s
            ScalarType old_alpha = alpha;
            ScalarType old_phi = current_phi;
            ScalarType old_dphi = dphi;

            //Cubic extrapolation to chose a new value of ai
            ScalarType xmin = alpha + 0.01*(alpha-alpham1);
            ScalarType xmax = 10*alpha;
            alpha = cubicmin(alpham1,alpha,last_phi,current_phi,dphim1,dphi,xmin,xmax);
            if(std::abs(alpha-xmin) < 1e-4 || std::abs(alpha-xmax) < 1e-4)
                alpha=(xmin+xmax)/2;
            alpham1 = old_alpha;
            last_phi = old_phi;
            dphim1 = old_dphi;
        }
        res.best_alpha = alpha;
        res.has_failed=true;
    }


private:
    /** parameter of the strong-wolfe powell conditions */
    ScalarType c1_;
    /** parameter of the strong-wolfe powell conditions */
    ScalarType c2_;
    /** temporary vector */
    VectorType x0_;


};




}

#endif
