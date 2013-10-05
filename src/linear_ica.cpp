/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * curveica - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "tests/benchmark-utils.hpp"

#include "curveica.h"

#include "umintl/check_grad.hpp"
#include "umintl/minimize.hpp"

#include "src/whiten.hpp"
#include "src/utils.hpp"
#include "src/backend.hpp"

#include "src/nonlinearities/extended_infomax.h"

#include "src/fastapprox.h"

#include <pmmintrin.h>

namespace curveica{


template<class ScalarType, class NonlinearityType>
struct ica_functor{
public:
    ica_functor(ScalarType const * data, std::size_t NF, std::size_t NC) : data_(data), NC_(NC), NF_(NF), nonlinearity_(NC,NF){
        is_first_ = true;

        ipiv_ =  new typename backend<ScalarType>::size_t[NC_+1];
        z1 = new ScalarType[NC_*NF_];

        phi = new ScalarType[NC_*NF_];

        phi_z1t = new ScalarType[NC_*NC_];
        dweights = new ScalarType[NC_*NC_];
        dbias = new ScalarType[NC_];
        W = new ScalarType[NC_*NC_];
        WLU = new ScalarType[NC_*NC_];
        b_ = new ScalarType[NC_];

        means_logp = new ScalarType[NC_];

        first_signs = new int[NC_];

        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType m2 = 0, m4 = 0;
            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType X = data_[c*NF_+f];
                m2 += std::pow(X,2);
                m4 += std::pow(X,4);
            }

            m2 = std::pow(1/(ScalarType)NF_*m2,2);
            m4 = 1/(ScalarType)NF_*m4;
            ScalarType k = m4/m2 - 3;
            first_signs[c] = (k+0.02>0)?1:-1;
        }
    }

    bool recompute_signs(){
        bool sign_change = false;

        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType m2 = 0, m4 = 0;
            ScalarType b = b_[c];
            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType X = z1[c*NF_+f];
                m2 += std::pow(X,2);
                m4 += std::pow(X,4);
            }

            m2 = std::pow(1/(ScalarType)NF_*m2,2);
            m4 = 1/(ScalarType)NF_*m4;
            ScalarType k = m4/m2 - 3;
            int new_sign = (k+0.02>0)?1:-1;
            sign_change |= (new_sign!=first_signs[c]);
            first_signs[c] = new_sign;
        }
        return sign_change;
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

        //std::cout << "2 - " << t.get() << std::endl;
        //phi = mean(mata.*abs(z2).^(mata-1).*sign(z2),2);
        nonlinearity_(z1,b_,first_signs,phi,means_logp);


        //std::cout << "3 - " << t.get() << std::endl;
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
        //std::cout << "4 - " << t.get() << std::endl;

        if(grad){

            //dbias = mean(phi,2)
            compute_mean(phi,NC_,NF_,dbias);
           // std::cout << "6 - " << t.get() << std::endl;

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
        //std::cout << "7 - " << t.get() << std::endl;


    }

private:
    ScalarType const * data_;
    int * first_signs;

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
    ScalarType* means_logp;

    NonlinearityType nonlinearity_;

    mutable bool is_first_;
};



options make_default_options(){
    options opt;
    opt.max_iter = 200;
    opt.verbosity_level = 2;
    return opt;
}


template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t DataNF, options const & opt){
    typedef typename umintl_backend<ScalarType>::type BackendType;
    umintl::minimizer<BackendType> minimizer;
    minimizer.direction = new umintl::quasi_newton<BackendType>(new umintl::lbfgs<BackendType>());
    //minimizer.direction = new umintl::conjugate_gradient<BackendType>(new umintl::polak_ribiere<BackendType>());
    //minimizer.direction = new umintl::quasi_newton<BackendType>(new umintl::bfgs<BackendType>());
    minimizer.verbosity_level = opt.verbosity_level;
    minimizer.max_iter = opt.max_iter;
    minimizer.stopping_criterion = new umintl::gradient_treshold<BackendType>(1e-6);
    std::size_t N = NC*NC + NC;
    std::size_t padsize = 4;
    std::size_t NF=(DataNF%padsize==0)?DataNF:(DataNF/padsize)*padsize;

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
    whiten<ScalarType>(NC, DataNF, NF, data,Sphere,white_data);
    detail::shuffle(white_data,NC,NF);

    ica_functor<ScalarType, extended_infomax_ica<ScalarType> > objective(white_data,NF,NC);

    do{
        minimizer(X,objective,X,N);
    }while(objective.recompute_signs());

    //Copies into datastructures
    std::memcpy(W, X,sizeof(ScalarType)*NC*NC);
    std::memcpy(b, X+NC*NC, sizeof(ScalarType)*NC);

    //out = W*Sphere*data;
    backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,2,data,DataNF,Sphere,NC,0,white_data,NF);
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

template void inplace_linear_ica<float>(float const * data, float * out, std::size_t NC, std::size_t NF, curveica::options const & opt);
template void inplace_linear_ica<double>(double const * data, double * out, std::size_t NC, std::size_t NF, curveica::options const & opt);

}

