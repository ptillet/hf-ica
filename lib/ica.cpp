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
#include "umintl/stopping_criterion/gradient_treshold.hpp"

#include "omp.h"

#include <stdlib.h>

namespace neo_ica{

inline int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

template<class T, class NonlinearityType>
struct ica_functor{
    typedef T * VectorType;

public:
    ica_functor(T const * data, int64_t NF, int64_t NC, options const &) : data_(data), NC_(NC), NF_(NF), nonlinearity_(NC,NF){
        is_first_ = true;

        ipiv_ =  new typename backend<T>::size_t[NC_+1];

        //NC*NF matrices
        Z = new T[NC_*NF_];
        RZ = new T[NC_*NF_];
        datasq_ = new T[NC_*NF_];

        //NC*NC matrices
        psixT = new T[NC_*NC_];
        phixT = new T[NC_*NC_];
        wmT = new T[NC_*NC_];
        W = new T[NC_*NC_];
        WLU = new T[NC_*NC_];
        V = new T[NC_*NC_];
        HV = new T[NC_*NC_];
        WinvV = new T[NC_*NC_];
        means_logp = new T[NC_];
        first_signs = new T[NC_];

        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0; j < NF_; ++j)
                datasq_[i*NF_+j] = data_[i*NF_+j]*data_[i*NF_+j];


        for(int64_t c = 0 ; c < NC_ ; ++c){
            T m2 = 0, m4 = 0;
            for(int64_t f = 0; f < NF_ ; f++){
                T X = data_[c*NF_+f];
                m2 += std::pow(X,2);
                m4 += std::pow(X,4);
            }
            m2 = std::pow(1/(T)NF_*m2,2);
            m4 = 1/(T)NF_*m4;
            T k = m4/m2 - 3;
            first_signs[c] = (T)((k+0.02>0)?1:-1);
        }
    }

    bool recompute_signs(T* x){
        bool sign_change = false;
        std::memcpy(W, x,sizeof(T)*NC_*NC_);
        backend<T>::gemm(NoTrans,NoTrans,NF_,NC_,NC_,1,data_,NF_,W,NC_,0,Z,NF_);

        for(int64_t c = 0 ; c < NC_ ; ++c){
            T m2 = 0, m4 = 0;
            //ScalarType b = b_[c];
            for(int64_t f = 0; f < NF_ ; f++){
                T X = Z[c*NF_+f];
                m2 += std::pow(X,2);
                m4 += std::pow(X,4);
            }

            m2 = std::pow(1/(T)NF_*m2,2);
            m4 = 1/(T)NF_*m4;
            T k = m4/m2 - 3;
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

        //Z = X*W
        std::memcpy(W, x,sizeof(T)*NC_*NC_);
        backend<T>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        //RZ = X*V
        std::memcpy(V, v,sizeof(T)*NC_*NC_);
        backend<T>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,V,NC_,0,RZ+offset,NF_);

        //Psi = dphi(Z).*RZ
        //Reuse Z's buffer because not needed anymore after and elementwise
        T* psi = Z;
        T* dphi = Z;
        nonlinearity_.compute_dphi(offset,sample_size,Z,first_signs,dphi);
        for(int64_t c = 0 ; c < NC_ ; ++c)
            for(int64_t f = offset; f < offset+sample_size ; ++f)
                psi[c*NF_+f] = dphi[c*NF_+f]*RZ[c*NF_+f];
        backend<T>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,psi+offset,NF_,0,psixT,NC_);


        //Variance = 1/(N-1)[psi.^2*(x.^2)' - 1/N*psi*x']
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = offset ; j < offset+sample_size; ++j)
                psi[i*NF_+j] = psi[i*NF_+j]*psi[i*NF_+j];
        backend<T>::gemm(Trans,NoTrans,NC_,NC_,sample_size,1,datasq_+offset,NF_,psi+offset,NF_,0,variance,NC_);
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0 ; j < NC_; ++j)
              variance[i*NC_+j] = (T)1/(sample_size-1)*(variance[i*NC_+j] - psixT[i*NC_+j]*psixT[i*NC_+j]/(T)sample_size);
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

        std::memcpy(W, x,sizeof(T)*NC_*NC_);

        //Z = X*W
        backend<T>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        //RZ = X*V
        std::memcpy(V, v,sizeof(T)*NC_*NC_);
        backend<T>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,V,NC_,0,RZ+offset,NF_);

        //Psi = dphi(Z).*RZ
        //Reuse Z's buffer because not needed anymore after and elementwise
        T* psi = Z;
        T* dphi = Z;
        nonlinearity_.compute_dphi(offset,sample_size,Z,first_signs,dphi);
        for(int64_t c = 0 ; c < NC_ ; ++c)
            for(int64_t f = offset; f < offset+sample_size ; ++f)
                psi[c*NF_+f] = dphi[c*NF_+f]*RZ[c*NF_+f];

        //HV = (inv(W)*V*inv(w))' + 1/n*Psi*X'
        std::memcpy(WLU,x,sizeof(T)*NC_*NC_);
        backend<T>::getrf(NC_,NC_,WLU,NC_,ipiv_);
        backend<T>::getri(NC_,WLU,NC_,ipiv_);
        backend<T>::gemm(Trans,Trans,NC_,NC_,NC_ ,1,WLU,NC_,V,NC_,0,WinvV,NC_);
        backend<T>::gemm(NoTrans,Trans,NC_,NC_,NC_ ,1,WinvV,NC_,WLU,NC_,0,HV,NC_);
        backend<T>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,psi+offset,NF_,0,psixT,NC_);

        //Copy back
        for(int64_t i = 0 ; i < NC_*NC_; ++i)
            Hv[i] = HV[i] + psixT[i]/(T)sample_size;
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

        std::memcpy(W, x,sizeof(T)*NC_*NC_);

        backend<T>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        T* phi = Z;
        nonlinearity_.compute_phi(offset,sample_size,Z,first_signs,phi);
        backend<T>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,phi+offset,NF_,0,phixT,NC_);

        //GradVariance = 1/(N-1)[phi.^2*(x.^2)' - 1/N*phi*x']
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = offset ; j < offset+sample_size; ++j)
                phi[i*NF_+j] = phi[i*NF_+j]*phi[i*NF_+j];
        backend<T>::gemm(Trans,NoTrans,NC_,NC_,sample_size,1,datasq_+offset,NF_,phi+offset,NF_,0,variance,NC_);
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0 ; j < NC_; ++j)
              variance[i*NC_+j] = (T)1/(sample_size-1)*(variance[i*NC_+j] - phixT[i*NC_+j]*phixT[i*NC_+j]/(T)sample_size);
    }

    /* Gradient */
    void operator()(VectorType const & x, T& value, VectorType & grad, umintl::value_gradient tag) const {
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
        std::memcpy(W, x,sizeof(T)*NC_*NC_);

        //Z = X*W;
        backend<T>::gemm(NoTrans,NoTrans,sample_size,NC_,NC_,1,data_+offset,NF_,W,NC_,0,Z+offset,NF_);

        //means_logp = mean(mata.*abs(Z).^(mata-1).*sign(Z),2);
        nonlinearity_.compute_means_logp(offset,sample_size,Z,first_signs,means_logp);

        //LU Decomposition
        std::memcpy(WLU,W,sizeof(T)*NC_*NC_);
        backend<T>::getrf(NC_,NC_,WLU,NC_,ipiv_);

        //H = log(abs(det(w))) + sum(means_logp);
        T logabsdet = 0;
        for(int64_t i = 0 ; i < NC_ ; ++i)
            logabsdet += std::log(std::abs(WLU[i*NC_+i]));
        T H = logabsdet;
        for(int64_t i = 0; i < NC_ ; ++i)
            H+=means_logp[i];

        //dweights = W^-T - 1/n*Phi*X'
        T* phi = Z;
        nonlinearity_.compute_phi(offset,sample_size,Z,first_signs,phi);
        backend<T>::gemm(Trans,NoTrans,NC_,NC_,sample_size ,1,data_+offset,NF_,phi+offset,NF_,0,phixT,NC_);
        backend<T>::getri(NC_,WLU,NC_,ipiv_);
        for(int64_t i = 0 ; i < NC_; ++i)
            for(int64_t j = 0 ; j < NC_; ++j)
                wmT[i*NC_+j] = WLU[j*NC_+i];

        //Reverse sign and copy
        value = -H;
        for(int64_t i = 0 ; i < NC_*NC_; ++i)
          grad[i] = - (wmT[i] - phixT[i]/sample_size);
    }

private:
    T const * data_;
    T * first_signs;

    int64_t NC_;
    int64_t NF_;


    typename backend<T>::size_t *ipiv_;


    T* Z ;
    T* RZ;

    T* phixT;
    T* psixT;

    T* datasq_;

    T* wmT;
    T* V;
    T* HV;
    T* WinvV;
    T* W;
    T* WLU;
    T* means_logp;

    NonlinearityType nonlinearity_;

    mutable bool is_first_;
};

template<class BackendType>
class stop_ica: public umintl::stopping_criterion<BackendType>
{
    typedef typename BackendType::ScalarType T;
public:
    stop_ica(T tol, int64_t NC): NC_(NC){}

    bool operator()(umintl::optimization_context<BackendType> & c)
    {
        using namespace std;
        T * W = new T[NC_*NC_];
        T * Wm1 = new T[NC_*NC_];
        T * tmp = new T[NC_*NC_];
        std::memcpy(W, c.x(),sizeof(T)*NC_*NC_);
        std::memcpy(Wm1, c.xm1(),sizeof(T)*NC_*NC_);
        //diff = max(abs(abs(diag(Wm1*W')) - 1))
        backend<T>::gemm(Trans,NoTrans,NC_,NC_,NC_ ,1,Wm1,NC_,W,NC_,0,tmp,NC_);
        T diff = 0;
        for(size_t i = 0 ; i < NC_ ; ++i)
            diff = max(diff, abs(abs(tmp[i*(NC_+1)]) - 1));
        delete[] W;
        delete[] Wm1;
        delete[] tmp;
        return diff < tol_;
    }
private:
    T tol_;
    int64_t NC_;
};

//lim = max(abs(abs(np.diag(fast_dot(W1, W.T))) - 1))

template<class ScalarType>
void ica(ScalarType const * data, ScalarType* Weights, ScalarType* Sphere, int64_t NC, int64_t DataNF, options const & conf){
    typedef typename umintl_backend<ScalarType>::type BackendType;
    typedef ica_functor<ScalarType, dist::sejnowski<ScalarType> > IcaFunctorType;

    options opt(conf);

    //Problem sizes
    int64_t padsize = 4;
    int64_t N = NC*NC;
    int64_t NF=(DataNF%padsize==0)?DataNF:(DataNF/padsize)*padsize;
    opt.fbatch=std::min(opt.fbatch, (size_t)NF);
    if(opt.fbatch==0)
        opt.fbatch=NF;

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
    minimizer.stopping_criterion = new umintl::parameter_change_threshold<BackendType>(1e-4);
    do{
        minimizer(X,objective,X,N);
    }while(objective.recompute_signs(X));

    //Copies into datastructures
    std::memcpy(Weights, X,sizeof(ScalarType)*NC*NC);

    delete[] X;
    delete[] white_data;
}

template void ica<float>(float const * data, float* Weights, float* Sphere, int64_t NC, int64_t NF, neo_ica::options const & opt);
template void ica<double>(double const * data, double* Weights, double* Sphere, int64_t NC, int64_t NF, neo_ica::options const & opt);

}

