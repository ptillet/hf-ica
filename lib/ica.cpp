/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "neo_ica/ica.h"
#include "neo_ica/dist/sejnowski.h"
#include "neo_ica/backend/backend.hpp"
#include "neo_ica/tools/mex.hpp"
#include "neo_ica/tools/shuffle.hpp"
#include "neo_ica/tools/whiten.hpp"

#include "umintl/debug.hpp"
#include "umintl/minimize.hpp"
#include "umintl/stopping_criterion/parameter_change_threshold.hpp"

#include "omp.h"

#include <stdlib.h>

namespace neo_ica{

inline int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

template<class _ScalarType, class NonlinearityType>
struct ica_functor{
    typedef _ScalarType ScalarType;
    typedef ScalarType * VectorType;

public:
    ica_functor(ScalarType const * data, int64_t NF, int64_t NC, options const & opt) : data_(data), NC_(NC), NF_(NF), nonlinearity_(NC,NF){
        is_first_ = true;

        ipiv_ =  new typename backend<ScalarType>::size_t[NC_+1];

        //NC*NF matrices
        Z = new ScalarType[NC_*NF_];
        RZ = new ScalarType[NC_*NF_];
        dphi = new ScalarType[NC_*NF_];
        phi = new ScalarType[NC_*NF_];
        psi = new ScalarType[NC_*NF_];
        datasq_ = new ScalarType[NC_*NF_];

        //NC*NC matrices
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

        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0; j < NF_; ++j)
                datasq_[i*NF_+j] = data_[i*NF_+j]*data_[i*NF_+j];


        int n_subgauss = 0;
        for(int64_t c = 0 ; c < NC_ ; ++c){
            ScalarType m2 = 0, m4 = 0;
            for(int64_t f = 0; f < NF_ ; f++){
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

        if(opt.verbosity>0){
            std::cout << "Number of subgaussian sources: " << n_subgauss << std::endl;
            std::cout << "Number of OMP Threads : " << omp_thread_count() << std::endl;
        }
    }

    bool recompute_signs(){
        bool sign_change = false;

        for(int64_t c = 0 ; c < NC_ ; ++c){
            ScalarType m2 = 0, m4 = 0;
            //ScalarType b = b_[c];
            for(int64_t f = 0; f < NF_ ; f++){
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
        //NC*NF matrices
        delete[] Z;
        delete[] RZ;
        delete[] dphi;
        delete[] phi;
        delete[] psi;
        delete[] datasq_;
        //NC*NC matrices
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

    /* Hessian-Vector product variance */
    void operator()(VectorType const & x, VectorType const & v, VectorType & variance, umintl::hv_product_variance tag) const{
        int64_t offset;
        int64_t sample_size;
        if(tag.model==umintl::DETERMINISTIC){
          offset = 0;
          sample_size = NF_;
        }
        else{
          offset = tag.offset;
          sample_size = tag.sample_size;
        }

        std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(WLU,x,sizeof(ScalarType)*NC_*NC_);
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        std::memcpy(V, v,sizeof(ScalarType)*NC_*NC_);
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,V,NC_,0,RZ+offset,NF_);

        //Psi = dphi(Z).*RZ
        nonlinearity_.compute_dphi(offset,sample_size,Z,first_signs,dphi);
        for(int64_t c = 0 ; c < NC_ ; ++c)
            for(int64_t f = offset; f < offset+sample_size ; ++f)
                psi[c*NF_+f] = dphi[c*NF_+f]*RZ[c*NF_+f];
        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,psi+offset,NF_,0,psixT,NC_);

        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = offset ; j < offset+sample_size; ++j)
                psi[i*NF_+j] = psi[i*NF_+j]*psi[i*NF_+j];

        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size,1,datasq_+offset,NF_,psi+offset,NF_,0,variance,NC_);

        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0 ; j < NC_; ++j)
              variance[i*NC_+j] = (ScalarType)1/(sample_size-1)*(variance[i*NC_+j] - psixT[i*NC_+j]*psixT[i*NC_+j]/(ScalarType)sample_size);
    }

    /* Hessian-Vector product */
    void operator()(VectorType const & x, VectorType const & v, VectorType & Hv, umintl::hessian_vector_product tag) const{
        int64_t offset;
        int64_t sample_size;
        if(tag.model==umintl::DETERMINISTIC){
          offset = 0;
          sample_size = NF_;
        }
        else{
          offset = tag.offset;
          sample_size = tag.sample_size;
        }

        std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(WLU,x,sizeof(ScalarType)*NC_*NC_);

        //Z = X*W
        backend<ScalarType>::getrf(NC_,NC_,WLU,NC_,ipiv_);
        backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        //RZ = X*V
        std::memcpy(V, v,sizeof(ScalarType)*NC_*NC_);
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,V,NC_,0,RZ+offset,NF_);

        //Psi = dphi(Z).*RZ
        nonlinearity_.compute_dphi(offset,sample_size,Z,first_signs,dphi);
        for(int64_t c = 0 ; c < NC_ ; ++c)
            for(int64_t f = offset; f < offset+sample_size ; ++f)
                psi[c*NF_+f] = dphi[c*NF_+f]*RZ[c*NF_+f];

        //HV = (inv(W)*V*inv(w))' + 1/n*Psi*X'
        backend<ScalarType>::gemm(Trans,Trans,NC_,NC_,NC_ ,1,WLU,NC_,V,NC_,0,WinvV,NC_);
        backend<ScalarType>::gemm(NoTrans,Trans,NC_,NC_,NC_ ,1,WinvV,NC_,WLU,NC_,0,HV,NC_);
        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,psi+offset,NF_,0,psixT,NC_);

        //Copy back
        for(int64_t i = 0 ; i < NC_*NC_; ++i)
            Hv[i] = HV[i] + psixT[i]/(ScalarType)sample_size;
    }

    /* Gradient variance */
    void operator()(VectorType const & x, VectorType & variance, umintl::gradient_variance tag){
        int64_t offset;
        int64_t sample_size;
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
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = offset ; j < offset+sample_size; ++j)
                phi[i*NF_+j] = phi[i*NF_+j]*phi[i*NF_+j];

        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size,1,datasq_+offset,NF_,phi+offset,NF_,0,variance,NC_);
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0 ; j < NC_; ++j)
              variance[i*NC_+j] = (ScalarType)1/(sample_size-1)*(variance[i*NC_+j] - phixT[i*NC_+j]*phixT[i*NC_+j]/(ScalarType)sample_size);
    }

    /* Gradient */
    void operator()(VectorType const & x, ScalarType& value, VectorType & grad, umintl::value_gradient tag) const {
        throw_if_mex_and_ctrl_c();

        int64_t offset;
        int64_t sample_size;
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

        //Z = X*W;
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        //means_logp = mean(mata.*abs(Z).^(mata-1).*sign(Z),2);
        nonlinearity_.compute_means_logp(offset,sample_size,Z,first_signs,means_logp);

        //LU Decomposition
        backend<ScalarType>::getrf(NC_,NC_,WLU,NC_,ipiv_);

        //det = prod(diag(WLU))
        ScalarType abs_det = 1;
        for(int64_t i = 0 ; i < NC_ ; ++i)
            abs_det*=std::abs(WLU[i*NC_+i]);

        //H = log(abs(det(w))) + sum(means_logp);
        ScalarType H = std::log(abs_det);
        for(int64_t i = 0; i < NC_ ; ++i)
            H+=means_logp[i];
        value = -H;

        /* Gradient */
        nonlinearity_.compute_phi(offset,sample_size,Z,first_signs,phi);

        //Grad = - (wmT - 1/S*phixT)
        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,phi+offset,NF_,0,phixT,NC_);

        //dweights = W^-T - 1/n*Phi*X'
        backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0 ; j < NC_; ++j)
                wmT[i*NC_+j] = WLU[j*NC_+i];

        //Copy back
        for(int64_t i = 0 ; i < NC_*NC_; ++i)
          grad[i] = - (wmT[i] - phixT[i]/(ScalarType)sample_size);
    }

private:
    ScalarType const * data_;
    int * first_signs;

    int64_t NC_;
    int64_t NF_;


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

template<class ScalarType>
void ica(ScalarType const * data, ScalarType* Weights, ScalarType* Sphere, int64_t NC, int64_t DataNF, options const & conf){
    typedef typename umintl_backend<ScalarType>::type BackendType;
    typedef ica_functor<ScalarType, dist::sejnowski<ScalarType> > IcaFunctorType;

    options opt(conf);

    //Problem sizes
    int64_t padsize = 4;
    int64_t N = NC*NC;
    int64_t NF=(DataNF%padsize==0)?DataNF:(DataNF/padsize)*padsize;
    if(opt.fbatch==0) opt.fbatch=NF;

    //Allocate
    ScalarType * white_data =  new ScalarType[NC*NF];
    ScalarType * X = new ScalarType[N];
    std::memset(X,0,N*sizeof(ScalarType));

    //Whiten Data
    whiten<ScalarType>(NC, DataNF, NF, data, Sphere, white_data);
    shuffle(white_data,NC,NF);
    IcaFunctorType objective(white_data,NF,NC,opt);

    //Initial guess W_0 = I
    for(int64_t i = 0 ; i < NC; ++i)
        X[i*(NC+1)] = 1;

    //Optimizer
    umintl::minimizer<BackendType> minimizer;
    minimizer.hessian_vector_product_computation = umintl::PROVIDED;
    minimizer.model = new umintl::dynamically_sampled<BackendType>(opt.rho,opt.fbatch,NF,opt.theta);
    minimizer.direction = new umintl::truncated_newton<BackendType>(umintl::tag::truncated_newton::STOP_HV_VARIANCE);
    minimizer.verbosity = opt.verbosity;
    minimizer.iter = opt.iter;
    minimizer.stopping_criterion = new umintl::parameter_change_threshold<BackendType>(1e-6);
    do{
        minimizer(X,objective,X,N);
    }while(objective.recompute_signs());

    //Copies into datastructures
    std::memcpy(Weights, X,sizeof(ScalarType)*NC*NC);

    delete[] X;
    delete[] white_data;
}

template void ica<float>(float const * data, float* Weights, float* Sphere, int64_t NC, int64_t NF, neo_ica::options const & opt);
template void ica<double>(double const * data, double* Weights, double* Sphere, int64_t NC, int64_t NF, neo_ica::options const & opt);

}

