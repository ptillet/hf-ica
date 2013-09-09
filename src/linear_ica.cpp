/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "Eigen/Dense"

#include "tests/benchmark-utils.hpp"

#include "fmincl/minimize.hpp"
#include "fmincl/backends/openblas.hpp"

#include "src/whiten.hpp"
#include "src/utils.hpp"


namespace parica{

template<class ScalarType>
struct ica_functor{
private:
    static const int alpha_sub = 4;
    static const int alpha_super = 1;
private:
    template <typename T>
    inline int sgn(T val) const {
        return (val>0)?1:-1;
    }
public:
    ica_functor(ScalarType const * data, std::size_t NC, std::size_t NF) : data_(data), NC_(NC), NF_(NF){
        ipiv_ =  new int[NC_+1];

        z1 = new ScalarType[NC_*NF_];
        phi = new ScalarType[NC_*NF_];
        phi_z1t = new ScalarType[NC_*NC_];
        dweights = new ScalarType[NC_*NC_];
        dbias = new ScalarType[NC_];
        W = new ScalarType[NC_*NC_];
        WLU = new ScalarType[NC_*NC_];
        b_ = new ScalarType[NC_];
        alpha = new ScalarType[NC_];
        means_logp = new ScalarType[NC_];
    }

    ~ica_functor(){
        delete ipiv_;
    }

    ScalarType operator()(ScalarType const * x, ScalarType ** grad) const {

        Timer t;
        t.start();

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(b_, x+NC_*NC_, sizeof(ScalarType)*NC_);



        //z1 = W*data_;
        blas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NC_,NF_,NC_,1,W,NC_,data_,NF_,0,z1,NF_);


        //z2 = z1 + b(:, ones(NF_,1));
        //kurt = (mean(z2.^2,2).^2) ./ mean(z2.^4,2) - 3
        //alpha = alpha_sub*(kurt<0) + alpha_super*(kurt>0)
        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType m2 = 0, m4 = 0;
            ScalarType b = b_[c];
            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType val = z1[c*NF_+f] + b;
                m2 += std::pow(val,2);
                m4 += std::pow(val,4);
            }
            m2 = std::pow(1/(ScalarType)NF_*m2,2);
            m4 = 1/(ScalarType)NF_*m4;
            ScalarType kurt = m4/m2 - 3;
            alpha[c] = alpha_sub*(kurt<0) + alpha_super*(kurt>=0);
        }


        //mata = alpha(:,ones(NF_,1));
        //logp = log(mata) - log(2) - gammaln(1./mata) - abs(z2).^mata;
        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType current = 0;
            ScalarType a = alpha[c];
            ScalarType b = b_[c];
            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType val = z1[c*NF_+f] + b;
                ScalarType fabs_val = std::fabs(val);
                current += (a==alpha_sub)?detail::compile_time_pow<alpha_sub>()(fabs_val):detail::compile_time_pow<alpha_super>()(fabs_val);
            }
            means_logp[c] = -1/(ScalarType)NF_*current + std::log(a) - std::log(2) - lgamma(1/a);
        }

        //H = log(abs(det(w))) + sum(means_logp);

        //LU Decomposition
        std::memcpy(WLU,W,sizeof(ScalarType)*NC_*NC_);
        blas_backend<ScalarType>::getrf(LAPACK_ROW_MAJOR,NC_,NC_,WLU,NC_,ipiv_);

        //det = prod(diag(WLU))
        ScalarType absdet = 1;
        for(std::size_t i = 0 ; i < NC_ ; ++i)
            absdet*=std::abs(WLU[i*NC_+i]);

        ScalarType H = std::log(absdet);
        for(std::size_t i = 0; i < NC_ ; ++i)
            H+=means_logp[i];

        if(grad){

            //phi = mean(mata.*abs(z2).^(mata-1).*sign(z2),2);
            for(unsigned int c = 0 ; c < NC_ ; ++c){
                ScalarType a = alpha[c];
                ScalarType b = b_[c];
                for(unsigned int f = 0 ; f < NF_ ; ++f){
                    ScalarType val = z1[c*NF_+f] + b;
                    ScalarType fabs_val = std::fabs(val);
                    ScalarType fabs_val_pow = (a==alpha_sub)?detail::compile_time_pow<alpha_sub-1>()(fabs_val):detail::compile_time_pow<alpha_super-1>()(fabs_val);
                    phi[c*NF_+f] = a*fabs_val_pow*sgn(val);
                }
            }

            //dbias = mean(phi,2)
            detail::mean(phi,NC_,NF_,dbias);

            /*dweights = -(eye(N) - 1/n*phi*z1')*inv(W)'*/

            //WLU = inv(W)
            blas_backend<ScalarType>::getri(LAPACK_ROW_MAJOR,NC_,WLU,NC_,ipiv_);

            //lhs = I(N,N) - 1/N*phi*z1')
            blas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasTrans,NC_,NC_,NF_ ,-1/(ScalarType)NF_,phi,NF_,z1,NF_,0,phi_z1t,NC_);
            for(std::size_t i = 0 ; i < NC_ ; ++i)
                phi_z1t[i*NC_+i] += 1;

            //dweights = -lhs*Winv'
            blas_backend<ScalarType>::gemm(CblasRowMajor, CblasNoTrans,CblasTrans,NC_,NC_,NC_,-1,phi_z1t,NC_,WLU,NC_,0,dweights,NC_);

            //Copy back
            std::memcpy(*grad, dweights,sizeof(ScalarType)*NC_*NC_);
            std::memcpy(*grad+NC_*NC_, dbias, sizeof(ScalarType)*NC_);
        }

        return -H;
    }

private:
    ScalarType const * data_;
    std::size_t NC_;
    std::size_t NF_;

    int *ipiv_;

    ScalarType* z1;
    ScalarType* phi;
    ScalarType* phi_z1t;
    ScalarType* dweights;
    ScalarType* dbias;
    ScalarType* W;
    ScalarType* WLU;
    ScalarType* b_;
    ScalarType* alpha;
    ScalarType* means_logp;
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

    size_t NC = data.rows();
    size_t NF = data.cols();
    std::size_t N = NC*NC + NC;

    ScalarType * data_copy = new ScalarType[NC*NF];
    ScalarType * W = new ScalarType[NC*NC];
    ScalarType * b = new ScalarType[NC];
    ScalarType * S = new ScalarType[N];
    ScalarType * X = new ScalarType[N]; std::memset(X,0,N*sizeof(ScalarType));
    ScalarType * white_data = new ScalarType[NC*NF];

    std::memcpy(data_copy,data.data(),NC*NF*sizeof(ScalarType));

    //Optimization Vector

    //Solution vector
    //Initial guess
    for(unsigned int i = 0 ; i < NC; ++i)
        X[i*(NC+1)] = 1;
    for(unsigned int i = NC*NC ; i < NC*(NC+1) ; ++i)
        X[i] = 0;

    //Whiten Data
    whiten<ScalarType>(NC, NF, data_copy,white_data);

    ica_functor<ScalarType> fun(white_data,NC,NF);

//    fmincl::utils::check_grad(fun,X);
    fmincl::minimize<fmincl::backend::OpenBlasTypes<ScalarType> >(S,fun,X,N,options);


    //Copies into datastructures
    std::memcpy(W, S,sizeof(ScalarType)*NC*NC);
    std::memcpy(b, S+NC*NC, sizeof(ScalarType)*NC);

    //out = W*white_data;
    blas_backend<ScalarType>::gemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NC,NF,NC,1,W,NC,white_data,NF,0,out.data(),NF);
    for(std::size_t c = 0 ; c < NC ; ++c){
        ScalarType val = b[c];
        for(std::size_t f = 0 ; f < NF ; ++f){
            out.data()[c*NF+f] += val;
        }
    }

    delete[] data_copy;
    delete[] W;
    delete[] b;
    delete[] S;
    delete[] X;
    delete[] white_data;

}


typedef result_of::internal_matrix_type<double>::type MatD;
typedef result_of::internal_matrix_type<float>::type MatF;
template void inplace_linear_ica<MatD,MatD>(MatD  const & data, MatD & out, fmincl::optimization_options const & options);
template void inplace_linear_ica<MatF,MatF>(MatF  const & data, MatF & out, fmincl::optimization_options const & options);

}

