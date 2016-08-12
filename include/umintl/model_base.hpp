#ifndef UMINTL_EVALUATION_POLICY_HPP
#define UMINTL_EVALUATION_POLICY_HPP

#include <cstddef>
#include "umintl/forwards.h"
#include "umintl/optimization_context.hpp"
#include <cmath>

namespace umintl{

/** @brief The model_base class
 *
 *  The optimization model can be either deterministic or stochastic.  The latter usually corresponds to expected losses
 * evaluated accross a large amount of data points
 */
template<class BackendType>
struct model_base{
    virtual ~model_base(){ }
    virtual bool update(optimization_context<BackendType> & context) = 0;
    virtual value_gradient get_value_gradient_tag() const = 0;
    virtual hessian_vector_product get_hv_product_tag() const = 0;
};

/** @brief The deterministic class
 *
 *  Assumes the function evaluation is the same at each call. In the case of expected losses, it means all the data-points
 * are always used.
 */
template<class BackendType>
struct deterministic : public model_base<BackendType> {
    bool update(optimization_context<BackendType> &){ return false; }
    value_gradient get_value_gradient_tag() const { return value_gradient(DETERMINISTIC,0,0); }
    hessian_vector_product get_hv_product_tag() const { return hessian_vector_product(DETERMINISTIC,0,0); }
};

template<class BackendType>
struct mini_batch : public model_base<BackendType> {
  public:
    mini_batch(size_t sample_size, size_t dataset_size) : sample_size_(std::min(sample_size,dataset_size)), offset_(0), dataset_size_(dataset_size){ }
    bool update(optimization_context<BackendType> &){
      offset_=(offset_+sample_size_)%dataset_size_;
      return false;
    }
    value_gradient get_value_gradient_tag() const { return value_gradient(STOCHASTIC,dataset_size_,0); }
    hessian_vector_product get_hv_product_tag() const { return hessian_vector_product(STOCHASTIC,sample_size_,offset_); }
private:
    size_t sample_size_;
    size_t offset_;
    size_t dataset_size_;
};

/** @brief the dynamically_sampled class
 *
 * Uses the dynamic sampled procedure from Byrd et al. (2012) :
 * "Sample Size Selection in Optimization Methods for Machine Learning"
 * Requires that the functor overloads :
 * void operator()(VectorType const & X, VectorType & variance, umintl::gradient_variance_tag tag)
 *
 * The parameter tag contains the information on the current offset and sample size
 */
template<class BackendType>
struct dynamically_sampled : public model_base<BackendType> {
  private:
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;

  public:
    dynamically_sampled(double r, size_t fbatch, size_t dataset_size, double theta = 0.5) : theta_(theta), r_(r), S(std::min(fbatch,dataset_size)), offset_(0), H_offset_(0), N(dataset_size){ }

    bool update(optimization_context<BackendType> & c){
      if(S==N){
        H_offset_=(H_offset_+(int)(r_*S))%(S - (int)(r_*S) + 1);
        return false;
      }
      else{
        VectorType var = BackendType::create_vector(c.N());
        c.fun().compute_gradient_variance(c.x(),var,gradient_variance(STOCHASTIC,S,offset_));

        //is_descent_direction = norm1(var)/S*[(N-S)/(N-1)] <= theta^2*norm2(grad)^2
        ScalarType nrm1var = BackendType::asum(c.N(),var);
        ScalarType nrm2grad = BackendType::nrm2(c.N(),c.g());
        //std::gradient_variance << nrm1var*scal << " " << std::pow(theta_,2)*std::pow(nrm2grad,2) << std::endl;
        bool is_descent_direction = (nrm1var/S <= (std::pow(theta_,2)*std::pow(nrm2grad,2)));

        //Update parameters
        //size_t old_S = S;
        if(is_descent_direction==false){
          S = (size_t)(nrm1var/std::pow(theta_*nrm2grad,2));
          S = std::min(S,N);
          if(S>N/2)
            S=N;
        }
        offset_=(offset_+S)%(N-S+1);

        if(is_descent_direction==false)
          H_offset_ = 0;
        else
          H_offset_=(H_offset_+S)%(S - (int)(r_*S) + 1);

        BackendType::delete_if_dynamically_allocated(var);
        return true;
      }
    }

    value_gradient get_value_gradient_tag() const {
      return value_gradient(STOCHASTIC,S,offset_);
    }

    hessian_vector_product get_hv_product_tag() const {
      return hessian_vector_product(STOCHASTIC,(size_t)(r_*S),H_offset_+offset_);
    }
private:
    double theta_;
    double r_;
    size_t S;
    size_t offset_;
    size_t H_offset_;
    size_t N;
};


}
#endif
