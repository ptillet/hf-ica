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

namespace parica{

template<class ScalarType>
struct ica_functor{
private:
    typedef typename result_of::data_storage<ScalarType>::type DataMatrixType;
private:
    template <typename T> int sgn(T val) const {
        return (T(0) < val) - (val < T(0));
    }
public:
    ica_functor(DataMatrixType const & data) : data_(data){ }

    double operator()(Eigen::VectorXd const & x, Eigen::VectorXd * grad) const {
        size_t nchans = data_.rows();
        size_t nframes = data_.cols();
        ScalarType cnframes = nframes;

        typename result_of::weights<ScalarType>::type W(nchans,nchans);
        Eigen::VectorXd b(nchans);

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W.data(), x.data(),sizeof(ScalarType)*nchans*nchans);
        std::memcpy(b.data(), x.data()+nchans*nchans, sizeof(ScalarType)*nchans);

        DataMatrixType z1 = W*data_;
        DataMatrixType z2 = z1; z2.colwise()+=b;

        Eigen::VectorXd alpha(nchans);

        for(unsigned int i = 0 ; i < nchans ; ++i){
            ScalarType m2 = 0, m4 = 0;
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
            DataMatrixType phi(nchans,nframes);

            for(unsigned int i = 0 ; i < nchans ; ++i){
                for(unsigned int j = 0 ; j < nframes ; ++j){
                    double a = alpha(i);
                    double z = z2(i,j);
                    phi(i,j) = a*std::pow(std::abs(z),(int)(a-1))*sgn(z);
                }
            }
            DataMatrixType phi_z1 = phi*z1.transpose();
            Eigen::VectorXd dbias = phi.rowwise().mean();
            DataMatrixType dweights(nchans, nchans);
            dweights = (DataMatrixType::Identity(nchans,nchans) - 1/cnframes*phi_z1);
            dweights = -dweights*W.transpose().inverse();
            std::memcpy(grad->data(), dweights.data(),sizeof(ScalarType)*nchans*nchans);
            std::memcpy(grad->data()+nchans*nchans, dbias.data(), sizeof(ScalarType)*nchans);
        }

        return -H;
    }

private:
    DataMatrixType const & data_;
};


fmincl::optimization_options make_default_options(){
    fmincl::optimization_options options;
    options.direction = new fmincl::quasi_newton(new fmincl::bfgs());
    options.max_iter = 200;
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

