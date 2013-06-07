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
        static void get_sphere(Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> & Cov, Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> & Sphere){
            Eigen::JacobiSVD<Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> > svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd svals = svd.singularValues();
            for(unsigned int i = 0 ; i < svals.size() ; ++i) svals[i] = 1/sqrt(svals[i]);
            Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic>  U = svd.matrixU();
            Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic>  V = U.transpose();
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
        Eigen::MatrixXd Cov = copy*copy.transpose();
        Cov = 1/(cnframes-1)*Cov;
        Eigen::MatrixXd Sphere(nchans,nchans);
        detail::get_sphere(Cov, Sphere);
        out = Sphere*data;
    }

}

template void clica::whiten<Eigen::MatrixXd>(Eigen::MatrixXd &, Eigen::MatrixXd &);
