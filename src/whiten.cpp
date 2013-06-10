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

        template<class NumericT>
        static void get_sphere(Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & Cov, Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & Sphere){
            typedef Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MAT;
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


    template<class MAT>
    void whiten(MAT & data, MAT & out){
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

}

template void clica::whiten< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &);
