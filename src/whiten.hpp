/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <iostream>

#include "Eigen/Dense"
#include "Eigen/SVD"

#include "result_of.hpp"

namespace parica{

    namespace detail{

        template<class M1, class M2>
        static void get_sphere(M1 & Cov, M2 & Sphere){
            Eigen::JacobiSVD<M1> svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd svals = svd.singularValues();
            for(unsigned int i = 0 ; i < svals.size() ; ++i) svals[i] = 1/sqrt(svals[i]);
            M1 U = svd.matrixU();
            M1 V = U.transpose();
            V = svals.asDiagonal()*V;
            Sphere = U*V;
            Sphere *= 2;
        }

    }


    template<class T, class U>
    void whiten(T & data, U & out){
        typedef typename T::Scalar ScalarType;
        typename result_of::data_storage<ScalarType>::type copy(data);
        unsigned int nchans = data.rows();
        unsigned int nframes = data.cols();
        double cnframes = nframes;
        Eigen::VectorXd means = 1/cnframes*copy.rowwise().sum();
        copy.colwise() -= means;
        typename result_of::weights<ScalarType>::type Cov = copy*copy.transpose();
        Cov = 1/(cnframes-1)*Cov;
        typename result_of::weights<ScalarType>::type Sphere(nchans,nchans);
        detail::get_sphere(Cov, Sphere);
        out = Sphere*data;
    }

}
