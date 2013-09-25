/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "tests/benchmark-utils.hpp"

#include "fmincl/check_grad.hpp"
#include "fmincl/minimize.hpp"

#include "src/whiten.hpp"
#include "src/utils.hpp"
#include "src/backend.hpp"


#include "fastapprox-0/fasthyperbolic.h"
#include "fastapprox-0/fastlog.h"

namespace parica{


template<class ScalarType>
struct ica_functor{
private:

    static const int alpha_sub = 4;
    static const int alpha_gauss = 2;
    static const int alpha_super = 1;
private:
    template <typename T>
    inline int sgn(T val) const {
        return (val>0)?1:-1;
    }

    static inline ScalarType fast_tanh(ScalarType x){
        ScalarType a = (((x*x+378)*x*x+17325)*x*x+135135)*x;
        ScalarType b = ((28*x*x+3150)*x*x+62370)*x*x+135135;
        return a/b;
    }

public:
    ica_functor(ScalarType const * data, std::size_t NF, std::size_t NC) : data_(data), NC_(NC), NF_(NF){
        ipiv_ =  new typename backend<ScalarType>::size_t[NC_+1];

        z1 = new ScalarType[NC_*NF_];

        phi = new ScalarType[NC_*NF_];


        phi_z1t = new ScalarType[NC_*NC_];
        dweights = new ScalarType[NC_*NC_];
        dbias = new ScalarType[NC_];
        W = new ScalarType[NC_*NC_];
        WLU = new ScalarType[NC_*NC_];
        b_ = new ScalarType[NC_];
        kurt = new ScalarType[NC_];
        means_logp = new ScalarType[NC_];
    }

    ~ica_functor(){
        delete[] ipiv_;

        delete[] z1;
        delete[] phi;
        delete[] phi_z1t;
        delete[] dweights;
        delete[] dbias;
        delete[] W;
        delete[] WLU;
        delete[] b_;
        delete[] kurt;
        delete[] means_logp;
    }

    void operator()(ScalarType const * x, ScalarType* value, ScalarType ** grad) const {
        Timer t;
        t.start();

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(b_, x+NC_*NC_, sizeof(ScalarType)*NC_);


        //z1 = W*data_;
        backend<ScalarType>::gemm(NoTrans,NoTrans,NF_,NC_,NC_,1,data_,NF_,W,NC_,0,z1,NF_);

        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType m2 = 0, m4 = 0;
            ScalarType b = b_[c];

            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType X = z1[c*NF_+f] + b;
                m2 += std::pow(X,2);
                m4 += std::pow(X,4);
            }

            m2 = std::pow(1/(ScalarType)NF_*m2,2);
            m4 = 1/(ScalarType)NF_*m4;
            ScalarType k = m4/m2 - 3;
            kurt[c] = k+0.02;

        }

        //y = tanh(z1 + b(:, ones(MiniBatch_NF_,1)));

        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType current = 0;
            ScalarType k = kurt[c];
            ScalarType b = b_[c];
            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType z2 = z1[c*NF_+f] + b;
                ScalarType y = fasttanh(z2);
                ScalarType logp = 0;
                if(k<0){
                    logp = std::log(0.5) + std::log((std::exp(-0.5*std::pow(z2-1,2)) + std::exp(-0.5*std::pow(z2+1,2))));
                }
                else{
                    logp = fastlog(1 - y*y) - 0.5*z2*z2;
                }
                current+=logp;
            }
            means_logp[c] = 1/(ScalarType)NF_*current;
        }

        //H = log(abs(det(w))) + sum(means_logp);
        //LU Decomposition
        std::memcpy(WLU,W,sizeof(ScalarType)*NC_*NC_);
        backend<ScalarType>::getrf(NC_,NC_,WLU,NC_,ipiv_);
        //det = prod(diag(WLU))
        ScalarType absdet = 1;
        for(std::size_t i = 0 ; i < NC_ ; ++i)
            absdet*=std::abs(WLU[i*NC_+i]);
        ScalarType H = std::log(absdet);
        for(std::size_t i = 0; i < NC_ ; ++i)
            H+=means_logp[i];

        if(value){
            *value = -H;
        }

        if(grad){
            //phi = mean(mata.*abs(z2).^(mata-1).*sign(z2),2);
            for(unsigned int c = 0 ; c < NC_ ; ++c){
                ScalarType k = kurt[c];
                ScalarType b = b_[c];
                for(unsigned int f = 0 ; f < NF_ ; f++){
                    ScalarType z2 = z1[c*NF_+f] + b;
                    ScalarType y = fastertanh(z2);
                    phi[c*NF_+f] =(k<0)?(z2 - y):(z2 + 2*y);
                }
            }


            //dbias = mean(phi,2)
            detail::compute_mean(phi,NC_,NF_,dbias);

            /*dweights = -(eye(N) - 1/n*phi*z1')*inv(W)'*/
            //WLU = inv(W)
            backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
            //lhs = I(N,N) - 1/N*phi*z1')
            backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,NF_ ,-1/(ScalarType)NF_,z1,NF_,phi,NF_,0,phi_z1t,NC_);
            for(std::size_t i = 0 ; i < NC_; ++i)
                phi_z1t[i*NC_+i] += 1;
            //dweights = -lhs*Winv'
            backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,NC_,-1,WLU,NC_,phi_z1t,NC_,0,dweights,NC_);
            //Copy back
            std::memcpy(*grad, dweights,sizeof(ScalarType)*NC_*NC_);
            std::memcpy(*grad+NC_*NC_, dbias, sizeof(ScalarType)*NC_);
        }

    }

private:
    ScalarType const * data_;
    std::size_t NC_;
    std::size_t NF_;


    typename backend<ScalarType>::size_t *ipiv_;

    //MiniBatch
    ScalarType* z1;
    ScalarType* phi;

    //Mixing
    ScalarType* phi_z1t;
    ScalarType* dweights;
    ScalarType* dbias;
    ScalarType* W;
    ScalarType* WLU;
    ScalarType* b_;
    ScalarType* kurt;
    ScalarType* means_logp;

};


fmincl::optimization_options make_default_options(){
    fmincl::optimization_options options;
    options.direction = new fmincl::quasi_newton();
    options.max_iter = 200;
    options.verbosity_level = 2;
    options.line_search = new fmincl::strong_wolfe_powell(5);
    options.stopping_criterion = new fmincl::gradient_treshold(1e-6);
    return options;
}


template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options){
    typedef typename fmincl_backend<ScalarType>::type BackendType;

    std::size_t N = NC*NC + NC;

    ScalarType * Sphere = new ScalarType[NC*NC];
    ScalarType * W = new ScalarType[NC*NC];
    ScalarType * b = new ScalarType[NC];
    ScalarType * X = new ScalarType[N];
    std::memset(X,0,N*sizeof(ScalarType));
    ScalarType * white_data = new ScalarType[NC*NF];

    //Optimization Vector

    //Solution vector
    //Initial guess W_0 = I
    //b_0 = 0
    for(unsigned int i = 0 ; i < NC; ++i)
        X[i*(NC+1)] = 1;

    //Whiten Data
    whiten<ScalarType>(NC, NF, data,Sphere);

    //white_data = randperm(2*Sphere*data)
    backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,2,data,NF,Sphere,NC,0,white_data,NF);

    detail::shuffle(white_data,NC,NF);
    ica_functor<ScalarType> objective(white_data,NF,NC);
    fmincl::minimize<BackendType>(X,objective,X,N,options);


    //Copies into datastructures
    std::memcpy(W, X,sizeof(ScalarType)*NC*NC);
    std::memcpy(b, X+NC*NC, sizeof(ScalarType)*NC);

    //out = W*Sphere*data;
    backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,2,data,NF,Sphere,NC,0,white_data,NF);
    backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,1,white_data,NF,W,NC,0,out,NF);

    for(std::size_t c = 0 ; c < NC ; ++c){
        ScalarType val = b[c];
        for(std::size_t f = 0 ; f < NF ; ++f){
            out[c*NF+f] += val;
        }
    }

    delete[] W;
    delete[] b;
    delete[] X;
    delete[] white_data;

}

template void inplace_linear_ica<float>(float const * data, float * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);
template void inplace_linear_ica<double>(double const * data, double * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);

}

