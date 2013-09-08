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
        static void inv_sqrtm(typename result_of::internal_matrix_type<ScalarType>::type & in,typename result_of::internal_matrix_type<ScalarType>::type & out){
            typedef typename result_of::internal_matrix_type<ScalarType>::type MatrixType;
            typedef typename result_of::internal_vector_type<ScalarType>::type VectorType;
            Eigen::JacobiSVD<MatrixType> svd(in, Eigen::ComputeThinU | Eigen::ComputeThinV);
            VectorType svals = svd.singularValues();
            for(unsigned int i = 0 ; i < svals.size() ; ++i)
                svals[i] = 1/sqrt(svals[i]);
            out = svd.matrixU()*svals.asDiagonal()*svd.matrixU().transpose();
        }

        template<class ScalarType>
        static void get_sphere(typename result_of::internal_matrix_type<ScalarType>::type & cov,typename result_of::internal_matrix_type<ScalarType>::type & sphere){
            inv_sqrtm<ScalarType>(cov,sphere);
            sphere *= 2;
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
        //Cov = data_copy*data_copy.transpose()
        openblas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasTrans,nchans,nchans,nframes
                                           ,1,data_copy.data(),nframes,data_copy.data(),nframes,0,Cov.data(),nchans);
        Cov = 1/static_cast<ScalarType>(nframes-1)*Cov;
        MatrixType Sphere(nchans,nchans);
        detail::get_sphere<ScalarType>(Cov, Sphere);
        out = Sphere*data_copy;
    }

}
