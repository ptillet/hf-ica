/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <iostream>

#include "utils.hpp"

#include "parica.h"
#include "blas_backend.hpp"

namespace parica{

    namespace detail{

        template<class ScalarType>
        static void inv_sqrtm(std::size_t C, ScalarType * in, ScalarType * out){

            ScalarType * D = new ScalarType[C];
            ScalarType * UD = new ScalarType[C*C];

            //in = U
            blas_backend<ScalarType>::syevd(LAPACK_ROW_MAJOR,'V','U',C,in,C,D);

            //UD = U*diag(D)
            for (std::size_t j=0; j<C; ++j) {
              ScalarType lambda = 1/std::sqrt(D[j]);
              for (std::size_t i=0; i<C; ++i)
                  UD[i*C+j] = in[i*C+j]*lambda;
            }

            //out = UD*U^T
            blas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasTrans,C,C,C,1,UD,C,in,C,0,out,C);

            delete[] D;
            delete[] UD;
        }

        template<class ScalarType>
        void mean(ScalarType* A, std::size_t C, std::size_t N, ScalarType* x){
            for(std::size_t i = 0 ; i < C ;++i){
                ScalarType sum = 0;
                for(std::size_t j = 0 ; j < N ; ++j)
                    sum += A[i*N+j];
                x[i] = sum/(ScalarType)N;
            }
        }


        template<class ScalarType>
        void normalize(ScalarType* A, std::size_t C, std::size_t N){
            ScalarType * x = new ScalarType[C];

            mean(A,C,N,x);
            for(std::size_t i = 0 ; i < C ;++i)
                for(std::size_t j = 0 ; j < N ; ++j)
                    A[i*N+j] -= x[i];

            delete[] x;
        }

    }


    template<class ScalarType>
    void whiten(std::size_t nchans, std::size_t nframes, ScalarType * data_copy, ScalarType * out){
        ScalarType * Cov = new ScalarType[nchans*nchans];
        ScalarType * Sphere = new ScalarType[nchans*nchans];

        //data_copy -= mean(data_copy,2);
        detail::normalize(data_copy,nchans,nframes);
        //Cov = 1/(N-1)*data_copy*data_copy.transpose()
        ScalarType alpha = (ScalarType)(1)/(nframes-1);
        blas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasTrans,nchans,nchans,nframes,alpha,data_copy,nframes,data_copy,nframes,0,Cov,nchans);
        //Sphere = inverse(sqrtm(Cov))
        detail::inv_sqrtm<ScalarType>(nchans,Cov,Sphere);
        //out = 2*Sphere*data_copy;
        blas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nchans,nframes,nchans,2,Sphere,nchans,data_copy,nframes,0,out,nframes);

        delete[] Cov;
        delete[] Sphere;
    }

}
