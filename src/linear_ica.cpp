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

#include <pmmintrin.h>

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

    inline void fill_y() const;
    inline void compute_means_logp() const;


public:
    ica_functor(ScalarType const * data, std::size_t NF, std::size_t NC) : data_(data), NC_(NC), NF_((NF%4==0)?NF:(NF/4)*NF){
        ipiv_ =  new typename backend<ScalarType>::size_t[NC_+1];
        z1 = new ScalarType[NC_*NF_];
        y_ = new ScalarType[NC_*NF_];

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

        fill_y();

        compute_means_logp();

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
                    ScalarType y = y_[c*NF_+f];
                    phi[c*NF_+f] =(k<0)?(z2 - y):(z2 + 2*y);
                }
            }


            //dbias = mean(phi,2)
            compute_mean(phi,NC_,NF_,dbias);

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

    ScalarType* y_;

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

template<>
void ica_functor<float>::fill_y() const{

    for(unsigned int c = 0 ; c < NC_ ; ++c){
        const __m128 bias = _mm_set1_ps(b_[c]);
        for(unsigned int f = 0; f < NF_ ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);
            z2 = _mm_add_ps(z2,bias);
            __m128 y = vfasttanh(z2);
            _mm_store_ps(&y_[c*NF_+f],y);
        }
    }
}

template<>
void ica_functor<double>::fill_y() const{
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        double bias = b_[c];
        for(unsigned int f = 0; f < NF_ ; f++){
            y_[c*NF_+f] = std::tanh(z1[c*NF_+f]+bias);
        }
    }
}

template<>
void ica_functor<float>::compute_means_logp() const{
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        float k = kurt[c];
        const __m128 bias = _mm_set1_ps(b_[c]);
        for(unsigned int f = 0; f < NF_ ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);
            z2 = _mm_add_ps(z2,bias);
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 m0_5 = _mm_set1_ps(-0.5);
            if(k<0){
                const __m128 vln0_5 = _mm_set1_ps(-0.693147);

                __m128 a = _mm_sub_ps(z2,_1);
                a = _mm_mul_ps(a,a);
                a = _mm_mul_ps(m0_5,a);
                a = vfastexp(a);

                __m128 b = _mm_add_ps(z2,_1);
                b = _mm_mul_ps(b,b);
                b = _mm_mul_ps(m0_5,b);
                b = vfastexp(b);

                a = _mm_add_ps(a,b);
                a = vfastlog(a);

                a = _mm_add_ps(vln0_5,a);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(a));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(a,a)));
            }
            else{
                __m128 z22 = _mm_mul_ps(z2,z2);
                z22 = _mm_mul_ps(_mm_set1_ps(0.5),z22);

                __m128 y = _mm_load_ps(&y_[c*NF_+f]);
                y = _mm_mul_ps(y,y);
                y = _mm_sub_ps(_1,y);
                y = vfastlog(y);
                y = _mm_sub_ps(y,z22);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(y));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(y,y)));
            }
        }
        double sum;
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(double)NF_*sum;
    }
}

//template<>
//void ica_functor<double>::compute_means_logp() const{
//    for(unsigned int c = 0 ; c < NC_ ; ++c){
//        double current = 0;
//        double k = kurt[c];
//        double b = b_[c];
//        for(unsigned int f = 0; f < NF_ ; f++){
//            double z2 = z1[c*NF_+f] + b;
//            double y = y_[c*NF_+f];
//            float logp = 0;
//            if(k<0){
//                logp = std::log(0.5) + std::log((std::exp(-0.5*std::pow(z2-1,2)) + std::exp(-0.5*std::pow(z2+1,2))));
//            }
//            else{
//                logp = std::log(1 - y*y) - 0.5*z2*z2;
//            }
//            current+=logp/(double)NF_;
//        }
//        means_logp[c] = current;
//    }
//}

template<>
void ica_functor<double>::compute_means_logp() const{
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        float k = kurt[c];
        const __m128 bias = _mm_set1_ps(b_[c]);
        for(unsigned int f = 0; f < NF_ ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));
            z2 = _mm_add_ps(z2,bias);
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 m0_5 = _mm_set1_ps(-0.5);
            if(k<0){
                const __m128 vln0_5 = _mm_set1_ps(-0.693147);

                __m128 a = _mm_sub_ps(z2,_1);
                a = _mm_mul_ps(a,a);
                a = _mm_mul_ps(m0_5,a);
                a = vfastexp(a);

                __m128 b = _mm_add_ps(z2,_1);
                b = _mm_mul_ps(b,b);
                b = _mm_mul_ps(m0_5,b);
                b = vfastexp(b);

                a = _mm_add_ps(a,b);
                a = vfastlog(a);

                a = _mm_add_ps(vln0_5,a);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(a));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(a,a)));
            }
            else{
                __m128 z22 = _mm_mul_ps(z2,z2);
                z22 = _mm_mul_ps(_mm_set1_ps(0.5),z22);

                __m128d ylo = _mm_load_pd(&y_[c*NF_+f]);
                ylo = _mm_mul_pd(ylo,ylo);
                ylo = _mm_sub_pd(_mm_set1_pd(1),ylo);
                __m128d yhi = _mm_load_pd(&y_[c*NF_+f+2]);
                yhi = _mm_mul_pd(yhi,yhi);
                yhi = _mm_sub_pd(_mm_set1_pd(1),yhi);
                __m128 y = _mm_movelh_ps(_mm_cvtpd_ps(ylo), _mm_cvtpd_ps(yhi));

                y = vfastlog(y);
                y = _mm_sub_ps(y,z22);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(y));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(y,y)));
            }
        }
        double sum;
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(float)NF_*sum;
    }
}


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
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t DataNF, fmincl::optimization_options const & options){
    typedef typename fmincl_backend<ScalarType>::type BackendType;

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

    ica_functor<ScalarType> objective(white_data,NF,NC);
    fmincl::minimize<BackendType>(X,objective,X,N,options);


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

template void inplace_linear_ica<float>(float const * data, float * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);
template void inplace_linear_ica<double>(double const * data, double * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);

}

