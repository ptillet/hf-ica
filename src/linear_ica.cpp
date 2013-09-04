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
#include "fmincl/backends/eigen.hpp"

#include "whiten.hpp"
#include "parica.h"
#include "Eigen/Dense"

#include "utils.hpp"

#include "tests/benchmark-utils.hpp"

#include "cblas.h"

namespace parica{

template<class ScalarType>
struct ica_functor{
private:
    typedef typename result_of::internal_matrix_type<ScalarType>::type MatrixType;
    typedef typename result_of::internal_vector_type<ScalarType>::type VectorType;
    static const int alpha_sub = 4;
    static const int alpha_super = 1;
private:
    template <typename T>
    inline int sgn(T val) const {
        return (val>0)?1:-1;
    }
public:
    ica_functor(MatrixType const & data) : data_(data), nchans_(data.rows()), nframes_(data.cols()){
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

    ScalarType operator()(VectorType const & x, VectorType * grad) const {
        size_t nchans = data_.rows();
        size_t nframes = data_.cols();
        ScalarType casted_nframes = nframes;


        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W.data(), x.data(),sizeof(ScalarType)*nchans*nchans);
        std::memcpy(b_.data(), x.data()+nchans*nchans, sizeof(ScalarType)*nchans);
        gemm(1,W,data_,0,z1);


        for(unsigned int i = 0 ; i < nchans ; ++i){
            ScalarType m2 = 0, m4 = 0;
            ScalarType b = b_(i);
            for(unsigned int j = 0; j < nframes ; j++){
                ScalarType val = z1(i,j) + b;
                m2 += std::pow(val,2);
                m4 += std::pow(val,4);
            }
            m2 = std::pow(1/casted_nframes*m2,2);
            m4 = 1/casted_nframes*m4;
            ScalarType kurt = m4/m2 - 3;
            alpha(i) = alpha_sub*(kurt<0) + alpha_super*(kurt>=0);
        }

        for(unsigned int i = 0 ; i < nchans ; ++i){
            ScalarType current = 0;
            ScalarType a = alpha[i];
            ScalarType b = b_(i);
            for(unsigned int j = 0; j < nframes ; j++){
                ScalarType val = z1(i,j) + b;
                ScalarType fabs_val = std::fabs(val);
                current += (a==alpha_sub)?compile_time_pow<alpha_sub>()(fabs_val):compile_time_pow<alpha_super>()(fabs_val);
            }
            means_logp[i] = -1/casted_nframes*current + std::log(a) - std::log(2) - lgamma(1/a);
        }

        ScalarType detweights = W.determinant();
        ScalarType H = std::log(std::abs(detweights)) + means_logp.sum();

        if(grad){
            for(unsigned int i = 0 ; i < nchans ; ++i){
                ScalarType a = alpha(i);
                ScalarType b = b_(i);
                for(unsigned int j = 0 ; j < nframes ; ++j){
                    ScalarType val = z1(i,j) + b;
                    ScalarType fabs_val = std::fabs(val);
                    ScalarType fabs_val_pow = (a==alpha_sub)?compile_time_pow<alpha_sub-1>()(fabs_val):compile_time_pow<alpha_super-1>()(fabs_val);
                    phi(i,j) = a*fabs_val_pow*sgn(val);
                }
            }
            gemm(1,phi,z1.transpose(),0,phi_z1);
            dbias = phi.rowwise().mean();
            dweights = (MatrixType::Identity(nchans,nchans) - 1/casted_nframes*phi_z1);
            MatrixType dweights_copy = dweights;
            MatrixType Winv = W;
            inplace_inverse(Winv);
            gemm(1,dweights_copy,Winv.transpose(),0,dweights);
            dweights = -dweights;
            std::memcpy(grad->data(), dweights.data(),sizeof(ScalarType)*nchans*nchans);
            std::memcpy(grad->data()+nchans*nchans, dbias.data(), sizeof(ScalarType)*nchans);
        }

        return -H;
    }

private:
    MatrixType const & data_;
    std::size_t nchans_;
    std::size_t nframes_;
    mutable MatrixType z1;
    mutable MatrixType phi;
    mutable MatrixType phi_z1;
    mutable MatrixType dweights;
    mutable VectorType dbias;
    mutable MatrixType W;
    mutable VectorType b_;
    mutable VectorType alpha;
    mutable VectorType means_logp;
};


fmincl::optimization_options make_default_options(){
    fmincl::optimization_options options;
    options.direction = new fmincl::quasi_newton_tag();
    options.max_iter = 100;
    options.verbosity_level = 0;
    return options;
}


template<class DataType, class OutType>
void inplace_linear_ica(DataType const & data, OutType & out, fmincl::optimization_options const & options){
    typedef typename DataType::Scalar ScalarType;
    typedef typename result_of::internal_matrix_type<ScalarType>::type MatrixType;
    typedef typename result_of::internal_vector_type<ScalarType>::type VectorType;

    size_t nchans = data.rows();
    size_t nframes = data.cols();

    MatrixType data_copy(data);

    MatrixType W(nchans,nchans);
    VectorType b(nchans);

    //Optimization Vector
    VectorType X = VectorType::Zero(nchans*nchans + nchans);
    for(unsigned int i = 0 ; i < nchans; ++i) X[i*(nchans+1)] = 1;
    for(unsigned int i = nchans*nchans ; i < nchans*(nchans+1) ; ++i) X[i] = 0;


    //Whiten Data
    MatrixType white_data(nchans, nframes);
    whiten<ScalarType>(data_copy,white_data);

    ica_functor<ScalarType> fun(white_data);
//    fmincl::utils::check_grad(fun,X);
    VectorType S =  fmincl::minimize<fmincl::backend::EigenTypes<ScalarType> >(fun,X, options);


    //Copies into datastructures
    std::memcpy(W.data(), S.data(),sizeof(ScalarType)*nchans*nchans);
    std::memcpy(b.data(), S.data()+nchans*nchans, sizeof(ScalarType)*nchans);

    (*generic_gemm<ScalarType>::get_ptr())(CblasRowMajor,CblasNoTrans,CblasNoTrans,nchans,nframes,nchans,1,W.data(),nchans,white_data.data(),nframes,0,out.data(),nframes); //out = W*white_data;
    out.colwise() += b;
}


typedef result_of::internal_matrix_type<double>::type MatD;
typedef result_of::internal_matrix_type<float>::type MatF;
template void inplace_linear_ica<MatD,MatD>(MatD  const & data, MatD & out, fmincl::optimization_options const & options);
template void inplace_linear_ica<MatF,MatF>(MatF  const & data, MatF & out, fmincl::optimization_options const & options);

}

