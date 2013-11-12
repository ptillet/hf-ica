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
#include "umintl/stopping_criterion/parameter_change_threshold.hpp"

#include "src/whiten.hpp"
#include "src/utils.hpp"
#include "src/backend.hpp"

#include "src/nonlinearities/extended_infomax.h"


namespace curveica{


template<class ScalarType, class NonlinearityType>
struct ica_functor{
    typedef ScalarType * VectorType;
public:
    ica_functor(ScalarType const * data, std::size_t NF, std::size_t NC) : data_(data), NC_(NC), NF_(NF), nonlinearity_(NC,NF){
        is_first_ = true;

        ipiv_ =  new typename backend<ScalarType>::size_t[NC_+1];
        Z = new ScalarType[NC_*NF_];
        RZ = new ScalarType[NC_*NF_];

        phi = new ScalarType[NC_*NF_];
        psi = new ScalarType[NC_*NF_];

        dweights = new ScalarType[NC_*NC_];

        W = new ScalarType[NC_*NC_];
        WLU = new ScalarType[NC_*NC_];
        V = new ScalarType[NC_*NC_];
        HV = new ScalarType[NC_*NC_];
        WinvV = new ScalarType[NC_*NC_];

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

        delete[] Z;
        delete[] RZ;

        delete[] phi;
        delete[] psi;
        delete[] dweights;
        delete[] W;
        delete[] WLU;
        delete[] V;
        delete[] HV;
        delete[] WinvV;

        delete[] means_logp;
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

        std::memcpy(W, x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(WLU,x,sizeof(ScalarType)*NC_*NC_);
        std::memcpy(V, v,sizeof(ScalarType)*NC_*NC_);

        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);
        backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,V,NC_,0,RZ+offset,NF_);


        //Psi = dphi(Z).*RZ
        nonlinearity_.compute_dphi(offset,sample_size,Z,first_signs,psi);
        for(unsigned int c = 0 ; c < NC_ ; ++c)
            for(unsigned int f = offset; f < offset+sample_size ; ++f)
                psi[c*NF_+f] *= RZ[c*NF_+f];

        //HV = (inv(W)*V*inv(w))' + 1/n*Psi*X'
        backend<ScalarType>::getrf(NC_,NC_,WLU,NC_,ipiv_);
        backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
        backend<ScalarType>::gemm(Trans,Trans,NC_,NC_,NC_ ,1,WLU,NC_,V,NC_,0,WinvV,NC_);
        backend<ScalarType>::gemm(NoTrans,Trans,NC_,NC_,NC_ ,1,WinvV,NC_,WLU,NC_,0,HV,NC_);

        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1/(ScalarType)sample_size,data_+offset,NF_,psi+offset,NF_,1,HV,NC_);

        //Copy back
        for(std::size_t i = 0 ; i < NC_*NC_; ++i)
            Hv[i] = HV[i];
    }

    void operator()(VectorType const & x, VectorType const & grad, VectorType & variance, umintl::gradient_variance tag){
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
      std::memcpy(WLU, W,sizeof(ScalarType)*NC_*NC_);
      std::memcpy(dweights, grad,sizeof(ScalarType)*NC_*NC_);
      backend<ScalarType>::getrf(NC_,NC_,WLU,NC_,ipiv_);
      backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
      backend<ScalarType>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);
      nonlinearity_.compute_phi(offset,sample_size,Z,first_signs,phi);

      for(std::size_t i = 0 ; i < NC_; ++i){
          for(std::size_t j = offset ; j < offset+sample_size; ++j){
              phi[i*NF_+j] = phi[i*NF_+j]*phi[i*NF_+j];
              RZ[i*NF_+j] = data_[i*NF_+j]*data_[i*NF_+j];
          }
      }

      backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size,1,RZ+offset,NF_,phi+offset,NF_,0,variance,NC_);

      for(std::size_t i = 0 ; i < NC_; ++i){
          for(std::size_t j = 0 ; j < NC_; ++j){
            ScalarType m = dweights[i*NC_+j]+WLU[j*NC_+i];
            variance[i*NC_+j] = (ScalarType)1/(sample_size-1)*(variance[i*NC_+j] - sample_size*m*m);
          }
      }

    }

    void operator()(VectorType const & x, ScalarType& value, VectorType & grad, umintl::value_gradient tag) const {
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

        nonlinearity_.compute_phi(offset,sample_size,Z,first_signs,phi);
        //dweights = W^-T - 1/n*Phi*X'
        backend<ScalarType>::getri(NC_,WLU,NC_,ipiv_);
        for(std::size_t i = 0 ; i < NC_; ++i)
            for(std::size_t j = 0 ; j < NC_; ++j)
                dweights[i*NC_+j] = WLU[j*NC_+i];
        backend<ScalarType>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,-1/(ScalarType)sample_size,data_+offset,NF_,phi+offset,NF_,1,dweights,NC_);

        //Copy back
        for(std::size_t i = 0 ; i < NC_*NC_; ++i)
            grad[i] = -dweights[i];
    }

private:
    ScalarType const * data_;
    int * first_signs;

    std::size_t NC_;
    std::size_t NF_;


    typename backend<ScalarType>::size_t *ipiv_;

    //MiniBatch
    ScalarType* Z;
    ScalarType* RZ;
    ScalarType* phi;

    //Mixing
    ScalarType* psi;
    ScalarType* dweights;
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
    opt.max_iter = 100;
    opt.verbosity_level = 2;
    opt.optimization_method = LBFGS;
    return opt;
}


template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t DataNF, options const & opt, double* W, double* S){
    typedef typename umintl_backend<ScalarType>::type BackendType;
    typedef ica_functor<ScalarType, extended_infomax_ica<ScalarType> > IcaFunctorType;

    std::size_t N = NC*NC;
    std::size_t padsize = 4;
    std::size_t NF=(DataNF%padsize==0)?DataNF:(DataNF/padsize)*padsize;

    ScalarType * Sphere = new ScalarType[NC*NC];
    ScalarType * Weights = new ScalarType[NC*NC];
    //ScalarType * b = new ScalarType[NC];
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
    IcaFunctorType objective(white_data,NF,NC);


    umintl::minimizer<BackendType> minimizer;
    minimizer.hessian_vector_product_computation = umintl::PROVIDED;
    //minimizer.model = new umintl::deterministic<BackendType>();
    if(opt.optimization_method==SD){
        minimizer.model = new umintl::dynamically_sampled<BackendType>(0.1,4,NF,0.5);
        minimizer.direction = new umintl::steepest_descent<BackendType>();
    }
    else if(opt.optimization_method==LBFGS)
        minimizer.direction = new umintl::quasi_newton<BackendType>(new umintl::lbfgs<BackendType>(16));
    else if(opt.optimization_method==NCG)
        minimizer.direction = new umintl::conjugate_gradient<BackendType>(new umintl::polak_ribiere<BackendType>());
    else if(opt.optimization_method==BFGS)
        minimizer.direction = new umintl::quasi_newton<BackendType>(new umintl::bfgs<BackendType>());
    else if(opt.optimization_method==HESSIAN_FREE){
      minimizer.direction = new umintl::truncated_newton<BackendType>();
      minimizer.model = new umintl::semi_stochastic<BackendType>(10000,NF);
    }

    minimizer.verbosity_level = opt.verbosity_level;
    minimizer.max_iter = opt.max_iter;
    //minimizer.stopping_criterion = new umintl::parameter_change_threshold<BackendType>(1e-6);
    minimizer.stopping_criterion = new umintl::gradient_treshold<BackendType>(1e-6);
    //do{
        minimizer(X,objective,X,N);
    //}while(objective.recompute_signs());


    //Copies into datastructures
    std::memcpy(Weights, X,sizeof(ScalarType)*NC*NC);
    //std::memcpy(b, X+NC*NC, sizeof(ScalarType)*NC);

    //out = W*Sphere*data;
    backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,1,data,DataNF,Sphere,NC,0,white_data,NF);
    backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,1,white_data,NF,Weights,NC,0,out,NF);

    for(std::size_t i = 0 ; i < NC ; ++i){
        for(std::size_t j = 0 ; j < NC ; ++j){
            if(W)
                W[i*NC+j] = Weights[j*NC+i];
            if(S)
                S[i*NC+j] = Sphere[j*NC+i];
        }
    }

    delete[] Weights;
    //delete[] b;
    delete[] X;
    delete[] white_data;

}

template void inplace_linear_ica<float>(float const * data, float * out, std::size_t NC, std::size_t NF, curveica::options const & opt, double* Weights, double* Sphere);
template void inplace_linear_ica<double>(double const * data, double * out, std::size_t NC, std::size_t NF, curveica::options const & opt, double * Weights, double * Sphere);

}

