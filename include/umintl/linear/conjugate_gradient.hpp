/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_
#define UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_

#include <limits>
#include <cmath>

#include "umintl/tools/shared_ptr.hpp"

namespace umintl{

  namespace linear{

    namespace conjugate_gradient_detail{

    /** @brief Base class for a stopping criterion for the linear conjugate gradient */
      template<class BackendType>
      struct stopping_criterion{
        private:
          typedef typename BackendType::VectorType VectorType;
          typedef typename BackendType::ScalarType ScalarType;
        public:
          virtual ~stopping_criterion(){ }
          virtual void init(VectorType const & p0) = 0;
          virtual void update(VectorType const & dk) = 0;
          virtual bool operator()(ScalarType rsn) = 0;
      };

      /** @brief residual norm stopping criterion
      *
      *  Stops the Linear CG when the norm of the residual is below a threshold
      */
      template<class BackendType>
      struct residual_norm : public stopping_criterion<BackendType>{
        private:
          typedef typename BackendType::VectorType VectorType;
          typedef typename BackendType::ScalarType ScalarType;
        public:
          residual_norm(double eps = 1e-4) : eps_(eps){ }
          void init(VectorType const & ){ }
          void update(VectorType const & ){ }
          bool operator()(ScalarType rsn){ return std::sqrt(rsn) < eps_; }
        private:
          ScalarType eps_;
      };

      /** @brief Base class for a matrix-vector product computation within linear conjugate gradient
      *
      * For the CG procedure, the explicit knowledge of the matrix is unnecessary. It is only necessary to know how to compute the product between
      * this matrix and any vector, hence this class
      */
      template<class BackendType>
      struct compute_Ab{
          virtual ~compute_Ab(){ }
          virtual void operator()(std::size_t N, typename BackendType::VectorType const & b, typename BackendType::VectorType & res) = 0;
      };

      /** @brief symv product class */
      template<class BackendType>
      struct symv : public compute_Ab<BackendType>{
        private:
          typedef typename BackendType::MatrixType MatrixType;
          typedef typename BackendType::VectorType VectorType;
        public:
          symv(MatrixType const & A) : A_(A){ }
          void operator()(std::size_t N, VectorType const & b, VectorType & res)
          {
            BackendType::symv(N,1,A_,b,0,res);
          }
        private:
          MatrixType const & A_;
      };


    }

    /** @brief Base class for the linear conjugate gradient
    *
    * This is a slightly modified version of the CG algorithm. Indeed,
    * the procedure is stopped whenever a direction of neative curvature is found
    */
    template<class BackendType>
    struct conjugate_gradient{
      private:
        typedef typename BackendType::MatrixType MatrixType;
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::ScalarType ScalarType;
      public:
        enum return_code{
          SUCCESS,
          FAILURE,
          FAILURE_NON_POSITIVE_DEFINITE
        };

        struct optimization_result{
            return_code ret;
            std::size_t i;
        };

      private:
        void allocate_tmp(std::size_t N){
          best_x = BackendType::create_vector(N);
          r = BackendType::create_vector(N);
          p = BackendType::create_vector(N);
          Ap = BackendType::create_vector(N);
        }

        optimization_result clear_terminate(return_code ret, std::size_t i){
          BackendType::delete_if_dynamically_allocated(best_x);
          BackendType::delete_if_dynamically_allocated(r);
          BackendType::delete_if_dynamically_allocated(p);
          BackendType::delete_if_dynamically_allocated(Ap);
          optimization_result res;
          res.ret = ret;
          res.i = i;
          return res;
        }

      public:

        conjugate_gradient(std::size_t _max_iter
                          , conjugate_gradient_detail::compute_Ab<BackendType> * _compute_Ab
                          , conjugate_gradient_detail::stopping_criterion<BackendType> * _stop = new umintl::linear::conjugate_gradient_detail::residual_norm<BackendType>)
          : max_iter(_max_iter), compute_Ab(_compute_Ab), stop(_stop){ }


        optimization_result operator()(std::size_t N, VectorType const & x0, VectorType const & b, VectorType & x)
        {
          allocate_tmp(N);
          ScalarType nrm_b = BackendType::nrm2(N,b);
          ScalarType lambda = 0;

          //x = x0;
          BackendType::copy(N,x0,x);

          ScalarType nrm_x0 = BackendType::nrm2(N,x0);
          if(nrm_x0==0){
            //r = b
            BackendType::copy(N,b,r);
          }
          else{
            //r = b - Ax0
            (*compute_Ab)(N,x,r); //r = Ax
            BackendType::scale(N,-1,r); //r = -Ax
            BackendType::axpy(N,1,b,r); //r = b - Ax
          }

          //p = r;
          BackendType::copy(N,r,p);

          stop->init(p);

          ScalarType rso = BackendType::dot(N,r,r);

          for(std::size_t i = 0 ; i < max_iter ; ++i){
            (*compute_Ab)(N,p,Ap);
            BackendType::axpy(N,lambda*nrm_b,b,Ap);


             //Ap = A*p
            ScalarType pAp = BackendType::dot(N,p,Ap);

            if(pAp<0){
              BackendType::copy(N,best_x,x);
              return clear_terminate(FAILURE_NON_POSITIVE_DEFINITE,i);
            }
            else
              BackendType::copy(N,x,best_x);

            ScalarType alpha = rso/pAp; //alpha = rso/(p'*Ap)
            BackendType::axpy(N,alpha,p,x); //x = x + alpha*p
            BackendType::axpy(N,-alpha,Ap,r); //r = r - alpha*Ap

            stop->update(x);

            //ScalarType quadval = -0.5*(BackendType::dot(N,x,r) + BackendType::dot(N,x,b)); //quadval = -0.5*(x'r + x'b);

            ScalarType rsn = BackendType::dot(N,r,r);

            if((*stop)(rsn))
              return clear_terminate(SUCCESS,i);

            BackendType::scale(N,rsn/rso,p);//pk = r + rsn/rso*pk
            BackendType::axpy(N,1,r,p);
            rso = rsn;
          }
          return clear_terminate(FAILURE,max_iter);
        }

        std::size_t max_iter;
        tools::shared_ptr<linear::conjugate_gradient_detail::compute_Ab<BackendType> > compute_Ab;
        tools::shared_ptr<linear::conjugate_gradient_detail::stopping_criterion<BackendType> > stop;
      private:
        VectorType r;
        VectorType p;
        VectorType Ap;
        VectorType best_x;
    };

  }

}

#endif
