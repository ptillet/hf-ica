/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#define FMINCL_WITH_EIGEN
#include "fmincl/minimize.hpp"

#include "clica.h"

#include "Eigen/Dense"

namespace clica{

template<class NumericT>
struct ica_functor{
private:
    typedef Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic> MAT;
private:
    template <typename T> int sgn(T val) const {
        return (T(0) < val) - (val < T(0));
    }
public:
    ica_functor(MAT const & data) : data_(data){ }

    double operator()(Eigen::VectorXd const & x, Eigen::VectorXd * grad) const {
        size_t nchans = data_.rows();
        size_t nframes = data_.cols();
        NumericT cnframes = nframes;

        MAT W(nchans,nchans);
        Eigen::VectorXd b(nchans);

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W.data(), x.data(),sizeof(NumericT)*nchans*nchans);
        std::memcpy(b.data(), x.data()+nchans*nchans, sizeof(NumericT)*nchans);

        MAT z1 = W*data_;
        MAT z2 = z1; z2.colwise()+=b;

        Eigen::VectorXd alpha(nchans);
        for(unsigned int i = 0 ; i < nchans ; ++i){
            NumericT m2 = 0, m4 = 0;
            for(unsigned int j = 0; j < nframes ; ++j){
                m2 += std::pow(z2(i,j),2);
                m4 += std::pow(z2(i,j),4);
            }
            m2 = std::pow(1/cnframes*m2,2);
            m4 = 1/cnframes*m4;
            double kurt = m4/m2 - 3;
            alpha(i) = 4*(kurt<0) + 1*(kurt>=0);
        }

        Eigen::VectorXd means_logp(nchans);
        for(unsigned int i = 0 ; i < nchans ; ++i){
            double current = 0;
            double a = alpha[i];
            for(unsigned int j = 0; j < nframes ; ++j){
                current -= std::pow(std::fabs(z2(i,j)),(int)a);
            }
            means_logp[i] = 1/cnframes*current + std::log(a) - std::log(2) - lgamma(1/a);
        }

        double detweights = W.determinant();
        double H = std::log(std::abs(detweights)) + means_logp.sum();
        if(grad){
            MAT phi(nchans,nframes);
            for(unsigned int i = 0 ; i < nchans ; ++i){
                for(unsigned int j = 0 ; j < nframes ; ++j){
                    double a = alpha(i);
                    double z = z2(i,j);
                    phi(i,j) = a*std::pow(std::abs(z),(int)(a-1))*sgn(z);
                }
            }
            MAT phi_z1 = phi*z1.transpose();
            Eigen::VectorXd dbias = phi.rowwise().mean();
            MAT dweights(nchans, nchans);
            dweights = (MAT::Identity(nchans,nchans) - 1/cnframes*phi_z1);
            dweights = -dweights*W.transpose().inverse();
            std::memcpy(grad->data(), dweights.data(),sizeof(NumericT)*nchans*nchans);
            std::memcpy(grad->data()+nchans*nchans, dbias.data(), sizeof(NumericT)*nchans);
        }

        return -H;
    }

private:
    MAT const & data_;
};

template<class T, class U>
void inplace_linear_ica(T & data, U & out){
    typedef typename T::Scalar ScalarType;
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> MAT;

    size_t nchans = data.rows();
    size_t nframes = data.cols();

    MAT W(nchans,nchans);
    Eigen::VectorXd b(nchans);

    //Optimization Vector
    Eigen::VectorXd X = Eigen::VectorXd::Zero(nchans*nchans + nchans);
    for(unsigned int i = 0 ; i < nchans; ++i) X[i*(nchans+1)] = 1;
    for(unsigned int i = nchans*nchans ; i < nchans*(nchans+1) ; ++i) X[i] = 0;

    //Whiten Data
    MAT white_data(nchans, nframes);
    whiten(data,white_data);

    ica_functor<ScalarType> fun(white_data);
    fmincl::optimization_options options;

//    options.line_search = fmincl::strong_wolfe_powell(1e-4,0.2);
//    options.direction = fmincl::cg<fmincl::polak_ribiere, fmincl::no_restart>();
    options.direction = new fmincl::quasi_newton(new fmincl::bfgs());
    options.max_iter = 2000;
    options.verbosity_level = 2;

//    fmincl::utils::check_grad(fun,X);
    Eigen::VectorXd S =  fmincl::minimize(fun,X, options);

    //Copies into datastructures
    std::memcpy(W.data(), S.data(),sizeof(ScalarType)*nchans*nchans);
    std::memcpy(b.data(), S.data()+nchans*nchans, sizeof(ScalarType)*nchans);


    out = W*white_data;
    out.colwise() += b;

}

typedef Eigen::MatrixXd MatDType;
typedef Eigen::Map<MatDType> MapMatDType;

template void inplace_linear_ica<MatDType, MatDType>(MatDType &, MatDType &);
template void inplace_linear_ica<MatDType, MapMatDType >(MatDType &, MapMatDType&);
template void inplace_linear_ica<MapMatDType, MatDType >(MapMatDType &, MatDType&);
template void inplace_linear_ica<MapMatDType, MapMatDType >(MapMatDType &, MapMatDType&);

}

