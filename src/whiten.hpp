/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <iostream>

#include "Eigen/SVD"
#include "parica.h"
#include "utils.hpp"

namespace parica{

    namespace detail{

        template<class ScalarType>
        static void inv_sqrtm(std::size_t N, ScalarType * in, ScalarType * out){

            ScalarType * W = new ScalarType[N];
            ScalarType * UD = new ScalarType[N*N];

            openblas_backend<ScalarType>::syevd(LAPACK_ROW_MAJOR,'V','U',N,in,N,W);

            for (std::size_t j=0; j<N; ++j) {
              ScalarType lambda = 1/std::sqrt(W[j]);
              for (std::size_t i=0; i<N; ++i)
                  UD[i*N+j] = in[i*N+j]*lambda;
            }

            openblas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasTrans,N,N,N,1,UD,N,in,N,0,out,N);


            delete[] W;
            delete[] UD;

        }

    }


    template<class ScalarType>
    void whiten(typename result_of::internal_matrix_type<ScalarType>::type & data_copy, typename result_of::internal_matrix_type<ScalarType>::type & out){
        typedef typename result_of::internal_matrix_type<ScalarType>::type MatrixType;
        typedef typename result_of::internal_vector_type<ScalarType>::type VectorType;

        unsigned int nchans = data_copy.rows();
        unsigned int nframes = data_copy.cols();
        VectorType means = 1/static_cast<ScalarType>(nframes)*data_copy.rowwise().sum();
        data_copy.colwise() -= means;
        MatrixType Cov(nchans,nchans);
        MatrixType Sphere(nchans,nchans);

        //Cov = 1/(N-1)*data_copy*data_copy.transpose()
        openblas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasTrans,nchans,nchans,nframes
                                           ,static_cast<ScalarType>(1)/static_cast<ScalarType>(nframes-1),data_copy.data(),nframes,data_copy.data(),nframes,0,Cov.data(),nchans);

        //Sphere = inverse(sqrtm(Cov))
        detail::inv_sqrtm<ScalarType>(nchans,Cov.data(),Sphere.data());

        openblas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nchans,nframes,nchans
                                           ,2,Sphere.data(),nchans,data_copy.data(),nframes,0,out.data(),nframes);    }

}
