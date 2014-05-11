/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * DSHF-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifdef __MINGW32__
#include <stdlib.h>
#include <mm_malloc.h> /* not even needed */
#include <windows.h> /* not even needed */
#define ALLOC_ALIGN(size) _mm_malloc(size,16)
#define FREE_ALIGN(ptr) _mm_free(ptr)
#else
#define ALLOC_ALIGN(size) aligned_alloc(16, size)
#define FREE_ALIGN(ptr) free(ptr)
#endif

#include "tests/benchmark-utils.hpp"

#include "dshf_ica.h"

#include "umintl/debug.hpp"
#include "umintl/minimize.hpp"
#include "umintl/stopping_criterion/parameter_change_threshold.hpp"

#include "src/whiten.hpp"
#include "src/utils.hpp"
#include "src/backend.hpp"

#include "src/nonlinearities/extended_infomax.h"

#include "omp.h"

#include <stdlib.h>

namespace dshf_ica{


template<class _ScalarType, class NonlinearityType>
struct ica_functor{
    typedef _ScalarType ScalarType  __attribute__((aligned (16)));
    typedef ScalarType * VectorType;

    bool weights_have_changed(VectorType x) const {
      bool result = false;
      for(std::size_t i = 0; i < NC_*NC_ ; ++i){
        if(W[i]!=x[i])
          result = true;
      }
      return result;
    }

public:
    ica_functor(ScalarType const * data, std::size_t NF, std::size_t NC) : data_(data), NC_(NC), NF_(NF), nonlinearity_(NC,NF){
        is_first_ = true;

        ipiv_ =  new typename backend<ScalarType>::size_t[NC_+1];

        Z = (ScalarType*)ALLOC_ALIGN(NC_*NF_*sizeof(ScalarType));
        RZ = (ScalarType*)ALLOC_ALIGN(NC_*NF_*sizeof(ScalarType));
        dphi = (ScalarType*)ALLOC_ALIGN(NC_*NF_*sizeof(ScalarType));
        phi = (ScalarType*)ALLOC_ALIGN(NC_*NF_*sizeof(ScalarType));
        psi = (ScalarType*)ALLOC_ALIGN(NC_*NF_*sizeof(ScalarType));
        datasq_ = (ScalarType*)ALLOC_ALIGN(NC_*NF_*sizeof(ScalarType));

        psixT = new ScalarType[NC_*NC_];
        phixT = new ScalarType[NC_*NC_];
        wmT = new ScalarType[NC_*NC_];
        W = new ScalarType[NC_*NC_];
        WLU = new ScalarType[NC_*NC_];
        V = new ScalarType[NC_*NC_];
        HV = new ScalarType[NC_*NC_];
        WinvV = new ScalarType[NC_*NC_];
        means_logp = new ScalarType[NC_];

        first_signs = new int[NC_];

        for(std::size_t i = 0 ; i < NC_; ++i)
            for(std::size_t j = 0; j < NF_; ++j)
                datasq_[i*NF_+j] = data_[i*NF_+j]*data_[i*NF_+j];


        int n_subgauss = 0;
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
            if(first_signs[c] < 0) n_subgauss++;
        }
        std::cout << n_subgauss << std::endl;
    }

    bool recompute_signs(){
        bool sign_change = false;

        for(unsigned int c = 0 ; c < NC_ ; ++c){
            ScalarType m2 = 0, m4 = 0;
            //ScalarType b = b_[c];
            for(unsigned int f = 0; f < NF_ ; f++){
                ScalarType X = Z[c*NF_+f];
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

        FREE_ALIGN(Z);
        FREE_ALIGN(RZ);
        FREE_ALIGN(dphi);
        FREE_ALIGN(phi);
        FREE_ALIGN(psi);
        FREE_ALIGN(datasq_);

        delete[] psixT;
        delete[] phixT;
        delete[] wmT;
        delete[] V;
        delete[] HV;
        delete[] W;
        delete[] WLU;
        delete[] WinvV;
        delete[] means_logp;
    }

    void operator()(VectorType const & x, VectorType const & v, VectorType & variance, umintl::hv_product_variance tag) const{
        std::size_t offset;
        std::size_t sample_size;
        if(tag.model==umintl::DETERMINISTIC){
          offset = 0;
          sample_size = NF_;
        }
        else{
          offset = tag.offset;
          sample_size = tag.sample_size;
        }

        //if(weights_have_changed(x)){
          std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
          std::memcpy(WLU,x,sizeof(ScalarType)*NC_*NC_);
          backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);
        //}


        std::memcpy(V, v,sizeof(ScalarType)*NC_*NC_);
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,V,NC_,0,RZ+offset,NF_);

        //Psi = dphi(Z).*RZ
        nonlinearity_.compute_dphi(offset,sample_size,Z,first_signs,dphi);
        for(unsigned int c = 0 ; c < NC_ ; ++c)
            for(unsigned int f = offset; f < offset+sample_size ; ++f)
                psi[c*NF_+f] = dphi[c*NF_+f]*RZ[c*NF_+f];
        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,psi+offset,NF_,0,psixT,NC_);

        for(std::size_t i = 0 ; i < NC_; ++i){
            for(std::size_t j = offset ; j < offset+sample_size; ++j){
                psi[i*NF_+j] = psi[i*NF_+j]*psi[i*NF_+j];
            }
        }
        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size,1,datasq_+offset,NF_,psi+offset,NF_,0,variance,NC_);

        for(std::size_t i = 0 ; i < NC_; ++i){
            for(std::size_t j = 0 ; j < NC_; ++j){
              variance[i*NC_+j] = (ScalarType)1/(sample_size-1)*(variance[i*NC_+j] - psixT[i*NC_+j]*psixT[i*NC_+j]/(ScalarType)sample_size);
            }
        }
    }

    void operator()(VectorType const & x, VectorType const & v, VectorType & Hv, umintl::hessian_vector_product tag) const{
        std::size_t offset;
        std::size_t sample_size;
        if(tag.model==umintl::DETERMINISTIC){
          offset = 0;
          sample_size = NF_;
        }
        else{
          offset = tag.offset;
          sample_size = tag.sample_size;
        }

        //if(weights_have_changed(x)){
          std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
          std::memcpy(WLU,x,sizeof(ScalarType)*NC_*NC_);

          backend<ScalarType>::getrf(NC_,NC_,WLU,NC_,ipiv_);
          backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
          backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);
        //}

        std::memcpy(V, v,sizeof(ScalarType)*NC_*NC_);
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,V,NC_,0,RZ+offset,NF_);


        //Psi = dphi(Z).*RZ
        nonlinearity_.compute_dphi(offset,sample_size,Z,first_signs,dphi);
        for(unsigned int c = 0 ; c < NC_ ; ++c)
            for(unsigned int f = offset; f < offset+sample_size ; ++f)
                psi[c*NF_+f] = dphi[c*NF_+f]*RZ[c*NF_+f];

        //HV = (inv(W)*V*inv(w))' + 1/n*Psi*X'
        backend<ScalarType>::gemm(Trans,Trans,NC_,NC_,NC_ ,1,WLU,NC_,V,NC_,0,WinvV,NC_);
        backend<ScalarType>::gemm(NoTrans,Trans,NC_,NC_,NC_ ,1,WinvV,NC_,WLU,NC_,0,HV,NC_);

        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,psi+offset,NF_,0,psixT,NC_);

        //Copy back
        for(std::size_t i = 0 ; i < NC_*NC_; ++i)
            Hv[i] = HV[i] + psixT[i]/(ScalarType)sample_size;
    }

    void operator()(VectorType const & x, VectorType & variance, umintl::gradient_variance tag){
      std::size_t offset;
      std::size_t sample_size;
      if(tag.model==umintl::DETERMINISTIC){
        offset = 0;
        sample_size = NF_;
      }
      else{
        offset = tag.offset;
        sample_size = tag.sample_size;
      }

      std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);

      backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

      nonlinearity_.compute_phi(offset,sample_size,Z,first_signs,phi);
      backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,phi+offset,NF_,0,phixT,NC_);

      //GradVariance = 1/(N-1)[phi.^2*(x.^2)' - 1/N*phi*x']
      for(std::size_t i = 0 ; i < NC_; ++i){
          for(std::size_t j = offset ; j < offset+sample_size; ++j){
              phi[i*NF_+j] = phi[i*NF_+j]*phi[i*NF_+j];
          }
      }
      backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size,1,datasq_+offset,NF_,phi+offset,NF_,0,variance,NC_);
      for(std::size_t i = 0 ; i < NC_; ++i){
          for(std::size_t j = 0 ; j < NC_; ++j){
            variance[i*NC_+j] = (ScalarType)1/(sample_size-1)*(variance[i*NC_+j] - phixT[i*NC_+j]*phixT[i*NC_+j]/(ScalarType)sample_size);
          }
      }
    }

    void operator()(VectorType const & x, ScalarType& value, VectorType & grad, umintl::value_gradient tag) const {
        throw_if_mex_and_ctrl_c();

        std::size_t offset;
        std::size_t sample_size;
        if(tag.model==umintl::DETERMINISTIC){
          offset = 0;
          sample_size = NF_;
        }
        else{
          offset = tag.offset;
          sample_size = tag.sample_size;
        }

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(WLU,W,sizeof(ScalarType)*NC_*NC_);

        //z1 = W*data_;
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        //phi = mean(mata.*abs(z2).^(mata-1).*sign(z2),2);
        nonlinearity_.compute_means_logp(offset,sample_size,Z,first_signs,means_logp);

        //LU Decomposition
        backend<ScalarType>::getrf(NC_,NC_,WLU,NC_,ipiv_);

        //det = prod(diag(WLU))
        ScalarType absdet = 1;
        for(std::size_t i = 0 ; i < NC_ ; ++i){
            absdet*=std::abs(WLU[i*NC_+i]);
        }

        //H = log(abs(det(w))) + sum(means_logp);
        ScalarType H = std::log(absdet);
        for(std::size_t i = 0; i < NC_ ; ++i){
            H+=means_logp[i];
        }
        value = -H;

        /* Gradient */
        nonlinearity_.compute_phi(offset,sample_size,Z,first_signs,phi);

        //Grad = - (wmT - 1/S*phixT)
        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,phi+offset,NF_,0,phixT,NC_);

        //dweights = W^-T - 1/n*Phi*X'
        backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
        for(std::size_t i = 0 ; i < NC_; ++i)
            for(std::size_t j = 0 ; j < NC_; ++j)
                wmT[i*NC_+j] = WLU[j*NC_+i];

        //Copy back
        for(std::size_t i = 0 ; i < NC_*NC_; ++i)
          grad[i] = - (wmT[i] - phixT[i]/(ScalarType)sample_size);

    }

private:
    ScalarType const * data_;
    int * first_signs;

    std::size_t NC_;
    std::size_t NF_;


    typename backend<ScalarType>::size_t *ipiv_;


    ScalarType* Z ;
    ScalarType* RZ;

    ScalarType* dphi;

    ScalarType* phi;
    ScalarType* phixT;

    ScalarType* psi;
    ScalarType* psixT;

    ScalarType* datasq_;

    ScalarType* wmT;
    ScalarType* V;
    ScalarType* HV;
    ScalarType* WinvV;
    ScalarType* W;
    ScalarType* WLU;
    ScalarType* means_logp;

    NonlinearityType nonlinearity_;

    mutable bool is_first_;
};



options make_default_options(){
    options opt;
    opt.max_iter = 500;
    opt.omp_num_threads = 0;
    opt.verbosity_level = 2;
    opt.RS = 0.1;
    opt.S0 = 0;
    opt.theta = 0.5;
    return opt;
}

static int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}


template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType* Weights, ScalarType* Sphere, std::size_t NC, std::size_t DataNF, options const & optimization_options){
    typedef typename umintl_backend<ScalarType>::type BackendType;
    typedef ica_functor<ScalarType, extended_infomax_ica<ScalarType> > IcaFunctorType;

    options opt(optimization_options);

    std::size_t padsize = 4;

    std::size_t N = NC*NC;
    std::size_t NF=(DataNF%padsize==0)?DataNF:(DataNF/padsize)*padsize;
    ScalarType * white_data =  (ScalarType*)ALLOC_ALIGN(NC*NF*sizeof(ScalarType));

    ScalarType * X = new ScalarType[N];
    std::memset(X,0,N*sizeof(ScalarType));

    if(opt.omp_num_threads>0)
        omp_set_num_threads(opt.omp_num_threads);
    if(opt.verbosity_level>=1)
        std::cout << "Number of OMP Threads : " << omp_thread_count() << std::endl;

    //Optimization Vector

    //Solution vector
    //Initial guess W_0 = I
    for(unsigned int i = 0 ; i < NC; ++i)
        X[i*(NC+1)] = 1;

    //Whiten Data

    whiten<ScalarType>(NC, DataNF, NF, data,Sphere,white_data);
    detail::shuffle(white_data,NC,NF);
    IcaFunctorType objective(white_data,NF,NC);


    umintl::minimizer<BackendType> minimizer;
    minimizer.hessian_vector_product_computation = umintl::PROVIDED;
    //minimizer.model = new umintl::deterministic<BackendType>();
    if(opt.S0==0)
      opt.S0=NF;
    minimizer.model = new umintl::dynamically_sampled<BackendType>(opt.RS,opt.S0,NF,opt.theta);
    minimizer.direction = new umintl::truncated_newton<BackendType>(umintl::tag::truncated_newton::STOP_HV_VARIANCE);

    minimizer.verbosity_level = opt.verbosity_level;
    minimizer.max_iter = opt.max_iter;
    minimizer.stopping_criterion = new umintl::parameter_change_threshold<BackendType>(1e-6);
    //minimizer.stopping_criterion = new umintl::value_treshold<BackendType>(1e-4);
    //minimizer.stopping_criterion = new umintl::gradient_treshold<BackendType>(1e-5);
    do{
        minimizer(X,objective,X,N);
    }while(objective.recompute_signs());


    //Copies into datastructures
    std::memcpy(Weights, X,sizeof(ScalarType)*NC*NC);

    delete[] X;
    FREE_ALIGN(white_data);
}

template void inplace_linear_ica<float>(float const * data, float* Weights, float* Sphere, std::size_t NC, std::size_t NF, dshf_ica::options const & opt);
template void inplace_linear_ica<double>(double const * data, double* Weights, double* Sphere, std::size_t NC, std::size_t NF, dshf_ica::options const & opt);

}

