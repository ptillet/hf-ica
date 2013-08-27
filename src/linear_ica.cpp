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

#include "whiten.hpp"
#include "result_of.hpp"
#include "Eigen/Dense"

#include "utils.hpp"

#include "tests/benchmark-utils.hpp"

#include "cblas.h"

namespace parica{

template<class ScalarType>
struct ica_functor{
private:
    typedef typename result_of::data_storage<ScalarType>::type DataMatrixType;
    static const int alpha_sub = 4;
    static const int alpha_super = 1;
private:
    template <typename T>
    inline int sgn(T val) const {
        return (val>0)?1:-1;
    }
public:
    ica_functor(DataMatrixType const & data) : data_(data), nchans_(data.rows()), nframes_(data.cols()){
        z1.resize(nchans_,nframes_);
        phi.resize(nchans_,nframes_);
        phi_z1.resize(nchans_,nchans_);
        dweights.resize(nchans_,nchans_);
        dbias.resize(nchans_);
        W.resize(nchans_,nchans_);
        b_.resize(nchans_);
        alpha.resize(nchans_);
        means_logp.resize(nchans_);
    }

    double operator()(Eigen::VectorXd const & x, Eigen::VectorXd * grad) const {
        Timer t;

        size_t nchans = data_.rows();
        size_t nframes = data_.cols();
        ScalarType casted_nframes = nframes;

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W.data(), x.data(),sizeof(ScalarType)*nchans*nchans);
        std::memcpy(b_.data(), x.data()+nchans*nchans, sizeof(ScalarType)*nchans);
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans
                   ,nchans,nframes,nchans,1,W.data(),nchans,data_.data(),nframes,0,z1.data(),nframes);

        for(unsigned int i = 0 ; i < nchans ; ++i){
            ScalarType m2 = 0, m4 = 0;
            double b = b_(i);
            for(unsigned int j = 0; j < nframes ; j++){
                double val = z1(i,j) + b;
                m2 += std::pow(val,2);
                m4 += std::pow(val,4);
            }
            m2 = std::pow(1/casted_nframes*m2,2);
            m4 = 1/casted_nframes*m4;
            double kurt = m4/m2 - 3;
            alpha(i) = alpha_sub*(kurt<0) + alpha_super*(kurt>=0);
        }


        for(unsigned int i = 0 ; i < nchans ; ++i){
            double current = 0;
            double a = alpha[i];
            double b = b_(i);
            for(unsigned int j = 0; j < nframes ; j++){
                double val = z1(i,j) + b;
                double fabs_val = std::fabs(val);
                current += (a==alpha_sub)?compile_time_pow<alpha_sub>()(fabs_val):compile_time_pow<alpha_super>()(fabs_val);
            }
            means_logp[i] = -1/casted_nframes*current + std::log(a) - std::log(2) - lgamma(1/a);
        }

        double detweights = W.determinant();
        double H = std::log(std::abs(detweights)) + means_logp.sum();

        if(grad){
            for(unsigned int i = 0 ; i < nchans ; ++i){
                double a = alpha(i);
                double b = b_(i);
                for(unsigned int j = 0 ; j < nframes ; ++j){
                    double val = z1(i,j) + b;
                    double fabs_val = std::fabs(val);
                    double fabs_val_pow = (a==alpha_sub)?compile_time_pow<alpha_sub-1>()(fabs_val):compile_time_pow<alpha_super-1>()(fabs_val);
                    phi(i,j) = a*fabs_val_pow*sgn(val);
                }
            }
            t.start();
            cblas_dgemm(CblasRowMajor, CblasNoTrans,CblasTrans
                       ,nchans,nchans,nframes,1,phi.data(),nframes,z1.data(),nframes,0,phi_z1.data(),nchans);
            dbias = phi.rowwise().mean();
            dweights = (DataMatrixType::Identity(nchans,nchans) - 1/casted_nframes*phi_z1);
            dweights = -dweights*W.transpose().inverse();
            std::memcpy(grad->data(), dweights.data(),sizeof(ScalarType)*nchans*nchans);
            std::memcpy(grad->data()+nchans*nchans, dbias.data(), sizeof(ScalarType)*nchans);
        }
        return -H;
    }

private:
    DataMatrixType const & data_;
    std::size_t nchans_;
    std::size_t nframes_;
    mutable DataMatrixType z1;
    mutable DataMatrixType phi;
    mutable DataMatrixType phi_z1;
    mutable DataMatrixType dweights;
    mutable Eigen::VectorXd dbias;
    mutable typename result_of::weights<ScalarType>::type W;
    mutable Eigen::VectorXd b_;
    mutable Eigen::VectorXd alpha;
    mutable Eigen::VectorXd means_logp;
};


fmincl::optimization_options make_default_options(){
    fmincl::optimization_options options;
    options.direction = new fmincl::quasi_newton();
    options.max_iter = 100;
    options.verbosity_level = 0;
    return options;
}


template<class T, class U>
void inplace_linear_ica(T & data, U & out, fmincl::optimization_options const & options){
    typedef typename T::Scalar ScalarType;

    size_t nchans = data.rows();
    size_t nframes = data.cols();

    typename result_of::weights<ScalarType>::type W(nchans,nchans);
    Eigen::VectorXd b(nchans);

    //Optimization Vector
    Eigen::VectorXd X = Eigen::VectorXd::Zero(nchans*nchans + nchans);
    for(unsigned int i = 0 ; i < nchans; ++i) X[i*(nchans+1)] = 1;
    for(unsigned int i = nchans*nchans ; i < nchans*(nchans+1) ; ++i) X[i] = 0;

    //Whiten Data
    typename result_of::data_storage<ScalarType>::type white_data(nchans, nframes);
    whiten(data,white_data);

    ica_functor<ScalarType> fun(white_data);
//    fmincl::utils::check_grad(fun,X);
    Eigen::VectorXd S =  fmincl::minimize(fun,X, options);

    //Copies into datastructures
    std::memcpy(W.data(), S.data(),sizeof(ScalarType)*nchans*nchans);
    std::memcpy(b.data(), S.data()+nchans*nchans, sizeof(ScalarType)*nchans);


    out = W*white_data;
    out.colwise() += b;

}

//Double col-major
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatXdc;
typedef Eigen::Map<MatXdc> MapMatXdc;
template void inplace_linear_ica<MatXdc, MatXdc>(MatXdc &, MatXdc &, fmincl::optimization_options const & );
template void inplace_linear_ica<MatXdc, MapMatXdc >(MatXdc &, MapMatXdc&, fmincl::optimization_options const & );
template void inplace_linear_ica<MapMatXdc, MatXdc >(MapMatXdc &, MatXdc&, fmincl::optimization_options const & );
template void inplace_linear_ica<MapMatXdc, MapMatXdc >(MapMatXdc &, MapMatXdc&, fmincl::optimization_options const & );

//Double row-major
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXdr;
typedef Eigen::Map<MatXdr> MapMatXdr;
template void inplace_linear_ica<MatXdr, MatXdr>(MatXdr &, MatXdr &, fmincl::optimization_options const & );
template void inplace_linear_ica<MatXdr, MapMatXdr >(MatXdr &, MapMatXdr&, fmincl::optimization_options const & );
template void inplace_linear_ica<MapMatXdr, MatXdr >(MapMatXdr &, MatXdr&, fmincl::optimization_options const & );
template void inplace_linear_ica<MapMatXdr, MapMatXdr >(MapMatXdr &, MapMatXdr&, fmincl::optimization_options const & );


}

