/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "tests/benchmark-utils.hpp"


#include "fmincl/minimize.hpp"

#include "src/whiten.hpp"
#include "src/utils.hpp"
#include "src/backend.hpp"


namespace parica{


template<class ScalarType>
struct ica_functor{
private:

    static const int alpha_sub = 4;
    static const int alpha_gauss = 2;
    static const int alpha_super = 1;
    static const int alpha_vsuper = 0.5;
private:
    template <typename T>
    inline int sgn(T val) const {
        return (val>0)?1:-1;
    }
public:
    ica_functor(ScalarType const * data, std::size_t NC, std::size_t NF) : data_(data), NC_(NC), NF_(NF){
        ipiv_ =  new typename backend<ScalarType>::size_t[NC_+1];

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
        delete[] ipiv_;

        delete[] z1;
        delete[] phi;
        delete[] phi_z1t;
        delete[] dweights;
        delete[] dbias;
        delete[] W;
        delete[] WLU;
        delete[] b_;
        delete[] alpha;
        delete[] means_logp;
    }


    ScalarType operator()(ScalarType const * x, ScalarType ** grad) const {

        Timer t;
        t.start();

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(b_, x+NC_*NC_, sizeof(ScalarType)*NC_);

        //z1 = W*data_;
        backend<ScalarType>::gemm(NoTrans,NoTrans,NF_,NC_,NC_,1,data_,NF_,W,NC_,0,z1,NF_);
        //z2 = z1 + b(:, ones(NF_,1));
        //kurt = (mean(z2.^2,2).^2) ./ mean(z2.^4,2) - 3
        //alpha = alpha_sub*(kurt<0) + alpha_super*(kurt>0)

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
            ScalarType kurt = m4/m2 - 3;
            ScalarType eps = 0.05;
            if(std::fabs(kurt) < eps)
                alpha[c]=alpha_gauss;
            else if(kurt<=-eps)
                alpha[c]=alpha_sub;
            else if(kurt>=eps)
                alpha[c]=alpha_super;
        }
        //mata = alpha(:,ones(NF_,1));
        //logp = log(mata) - log(2) - gammaln(1./mata) - abs(z2).^mata;
        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType current = 0;
            ScalarType a = alpha[c];
            ScalarType b = b_[c];
            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType X = z1[c*NF_+f] + b;
                if(a==alpha_gauss)
                    current+=std::pow(std::fabs(X),alpha_gauss);
                else if(a==alpha_sub)
                    current+=std::pow(std::fabs(X),alpha_sub);
                else if(a==alpha_super)
                    current+=std::pow(std::fabs(X),alpha_super);
            }
            means_logp[c] = -1/(ScalarType)NF_*current + std::log(a) - std::log(2) - lgamma(1/a);
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


        if(grad){

            //phi = mean(mata.*abs(z2).^(mata-1).*sign(z2),2);
            for(unsigned int c = 0 ; c < NC_ ; ++c){
                ScalarType a = alpha[c];
                ScalarType b = b_[c];
                for(unsigned int f = 0 ; f < NF_ ; f++){
                    ScalarType X = z1[c*NF_+f] + b;
                    ScalarType Xabs = std::fabs(X);
                    if(a==alpha_gauss)
                        phi[c*NF_+f] = a*std::pow(Xabs,alpha_gauss-1)*sgn(X);
                    else if(a==alpha_sub)
                        phi[c*NF_+f] = a*std::pow(Xabs,alpha_sub-1)*sgn(X);
                    else if(a==alpha_super)
                        phi[c*NF_+f] = a*std::pow(Xabs,alpha_super-1)*sgn(X);
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

        return -H;
    }

private:
    ScalarType const * data_;
    std::size_t NC_;
    std::size_t NF_;

    typename backend<ScalarType>::size_t *ipiv_;

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
    options.direction = new fmincl::quasi_newton();
    options.max_iter = 200;
    options.verbosity_level = 0;
    options.stopping_criterion = new fmincl::value_treshold(1e-4);
    return options;
}


template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options){
    std::size_t N = NC*NC + NC;

    ScalarType * data_copy = new ScalarType[NC*NF];
    ScalarType * W = new ScalarType[NC*NC];
    ScalarType * b = new ScalarType[NC];
    ScalarType * S = new ScalarType[N];
    ScalarType * X = new ScalarType[N];
    std::memset(X,0,N*sizeof(ScalarType));
    ScalarType * white_data = new ScalarType[NC*NF];
    std::memcpy(data_copy,data,NC*NF*sizeof(ScalarType));

    //Optimization Vector

    //Solution vector
    //Initial guess W_0 = I
    //b_0 = 0
    for(unsigned int i = 0 ; i < NC; ++i)
        X[i*(NC+1)] = 1;

    //Whiten Data
    whiten<ScalarType>(NC, NF, data_copy,white_data);

    ica_functor<ScalarType> fun(white_data,NC,NF);

    typedef typename fmincl_backend<ScalarType>::type FMinBackendType;
    //fmincl::utils::check_grad<FMinBackendType>(fun,X,N);
    fmincl::minimize<FMinBackendType>(S,fun,X,N,options);


    //Copies into datastructures
    std::memcpy(W, S,sizeof(ScalarType)*NC*NC);
    std::memcpy(b, S+NC*NC, sizeof(ScalarType)*NC);

    //out = W*white_data;
    backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,1,white_data,NF,W,NC,0,out,NF);
    for(std::size_t c = 0 ; c < NC ; ++c){
        ScalarType val = b[c];
        for(std::size_t f = 0 ; f < NF ; ++f){
            out[c*NF+f] += val;
        }
    }

    delete[] data_copy;
    delete[] W;
    delete[] b;
    delete[] S;
    delete[] X;
    delete[] white_data;

}

template void inplace_linear_ica<float>(float const * data, float * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);
template void inplace_linear_ica<double>(double const * data, double * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);

}

