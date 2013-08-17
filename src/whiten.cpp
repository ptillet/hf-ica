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

namespace clica{

    namespace detail{

        template<class MAT>
        static void get_sphere(MAT & Cov, MAT & Sphere){
            Eigen::JacobiSVD<MAT> svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd svals = svd.singularValues();
            for(unsigned int i = 0 ; i < svals.size() ; ++i) svals[i] = 1/sqrt(svals[i]);
            MAT U = svd.matrixU();
            MAT V = U.transpose();
            V = svals.asDiagonal()*V;
            Sphere = U*V;
            Sphere *= 2;
        }

    }


    template<class T, class U>
    void whiten(T & data, U & out){
        typedef typename T::Scalar ScalarType;
        typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> MAT;
        MAT copy(data);
        unsigned int nchans = data.rows();
        unsigned int nframes = data.cols();
        double cnframes = nframes;
        Eigen::VectorXd means = 1/cnframes*copy.rowwise().sum();

        copy.colwise() -= means;
        MAT Cov = copy*copy.transpose();
        Cov = 1/(cnframes-1)*Cov;
        MAT Sphere(nchans,nchans);
        detail::get_sphere(Cov, Sphere);
        out = Sphere*data;
    }

    typedef Eigen::MatrixXd MatDType;
    typedef Eigen::Map<MatDType> MapMatDType;

    template void whiten<MatDType, MatDType>(MatDType &, MatDType &);
    template void whiten<MatDType, MapMatDType >(MatDType &, MapMatDType&);
    template void whiten<MapMatDType, MatDType >(MapMatDType &, MatDType&);
    template void whiten<MapMatDType, MapMatDType >(MapMatDType &, MapMatDType&);
}
