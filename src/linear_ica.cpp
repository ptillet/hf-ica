/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#include "fmincl/minimize.hpp"

#include "clica.h"

#include "Eigen/Dense"

namespace clica{

template<class MAT>
struct ica_functor{
private:
    typedef double NumericT;
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

        Eigen::MatrixXd W(nchans,nchans);
        Eigen::VectorXd b(nchans);

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W.data(), x.data(),sizeof(NumericT)*nchans*nchans);
        std::memcpy(b.data(), x.data()+nchans*nchans, sizeof(NumericT)*nchans);

        Eigen::MatrixXd z1 = W*data_;
        Eigen::MatrixXd z2 = z1; z2.colwise()+=b;

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
                current -= std::pow(std::fabs(z2(i,j)),a);
            }
            means_logp[i] = 1/cnframes*current + std::log(a) - std::log(2) - lgamma(1/a);
        }

        double detweights = W.determinant();
        double H = std::log(std::abs(detweights)) + means_logp.sum();
        if(grad){
            Eigen::MatrixXd phi(nchans,nframes);
            for(unsigned int i = 0 ; i < nchans ; ++i){
                for(unsigned int j = 0 ; j < nframes ; ++j){
                    double a = alpha(i);
                    double z = z2(i,j);
                    phi(i,j) = a*std::pow(std::abs(z),a-1)*sgn(z);
                }
            }
            Eigen::MatrixXd phi_z1 = phi*z1.transpose();
            Eigen::VectorXd dbias = 1/cnframes*phi.rowwise().sum();
            Eigen::MatrixXd dweights(nchans, nchans);
            dweights = (Eigen::MatrixXd::Identity(nchans,nchans) - 1/cnframes*phi_z1);
            dweights = -dweights*W.transpose().inverse();
            std::memcpy(grad->data(), dweights.data(),sizeof(NumericT)*nchans*nchans);
            std::memcpy(grad->data()+nchans*nchans, dbias.data(), sizeof(NumericT)*nchans);
        }

        return -H;
    }

private:
    MAT const & data_;
};

template<class MAT>
void inplace_linear_ica(MAT & data, MAT & out){
    typedef double NumericT;

    size_t nchans = data.rows();
    size_t nframes = data.cols();

    Eigen::MatrixXd W(nchans,nchans);
    Eigen::VectorXd b(nchans);

    //Optimization Vector
    Eigen::VectorXd X = Eigen::VectorXd::Zero(nchans*nchans + nchans);
    for(unsigned int i = 0 ; i < nchans; ++i) X[i*(nchans+1)] = 1;
    for(unsigned int i = nchans*nchans ; i < nchans*(nchans+1) ; ++i) X[i] = 0;

    //Whiten Data
    Eigen::MatrixXd white_data(nchans, nframes);
    whiten(data,white_data);

    ica_functor<MAT> fun(white_data);
    fmincl::optimization_options options;

//    options.line_search = fmincl::strong_wolfe_powell(1e-4,0.2);
//    options.direction = fmincl::cg<fmincl::polak_ribiere, fmincl::no_restart>();
    options.direction = fmincl::quasi_newton<fmincl::bfgs>();
    options.line_search = fmincl::strong_wolfe_powell(1e-4,0.9);
    options.max_iter = 2000;
    options.verbosity_level = 2;

//    fmincl::utils::check_grad(fun,X);
    Eigen::VectorXd S =  fmincl::minimize(fun,X, options);

    //Copies into datastructures
    std::memcpy(W.data(), S.data(),sizeof(NumericT)*nchans*nchans);
    std::memcpy(b.data(), S.data()+nchans*nchans, sizeof(NumericT)*nchans);


    out = W*white_data;
    out.colwise() += b;

}

template void inplace_linear_ica<Eigen::MatrixXd>(Eigen::MatrixXd &, Eigen::MatrixXd &);

}

