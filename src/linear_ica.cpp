/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL
#define FMINCL_WITH_EIGEN

#include "tests/benchmark-utils.hpp"

#include "fmincl/minimize.hpp"

#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"

#include "clica.h"

#include "Eigen/Dense"

namespace clica{

template<class GMAT>
struct ica_functor{
private:
    typedef typename GMAT::value_type::value_type NumericT;
    typedef viennacl::vector<NumericT> GVEC;

    typedef viennacl::generator::matrix< GMAT> smat;
    typedef viennacl::generator::vector<NumericT> svec;

    typedef Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> CPU_MAT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic, 1> CPU_VEC;
public:
    ica_functor(GMAT const & data) : data_(data){ }

    double operator()(CPU_VEC const & x, CPU_VEC * grad) const {
        using namespace viennacl::generator;

        Timer t;

        size_t nchans = viennacl::traits::size1(data_);
        size_t nframes = viennacl::traits::size2(data_);

        NumericT cnframes = static_cast<NumericT>(nframes);


        CPU_MAT cpu_weights(nchans, nchans);
        CPU_VEC bias(nchans),cdbias(nchans), cmeans_logp(nchans);
        //Rerolls the variables into the appropriates datastructures
        std::memcpy(cpu_weights.data(), x.data(),sizeof(NumericT)*nchans*nchans);
        std::memcpy(bias.data(), x.data()+nchans*nchans, sizeof(NumericT)*nchans);

        //Creates GPU structures
        viennacl::matrix<NumericT> Gweights(nchans,nchans);
        GVEC gpu_bias(nchans);
        viennacl::copy(cpu_weights,Gweights);
        viennacl::copy(bias,gpu_bias);
        GMAT gpu_z1 = viennacl::linalg::prod(Gweights,data_);
        GMAT gpu_z2(nchans, nframes);
        GVEC gpu_m2(nchans);
        GVEC gpu_m4(nchans);
        GVEC gpu_kurt(nchans);
        GVEC gpu_alphas(nchans);
        GMAT gpu_logp(nchans, nframes);
        GVEC gpu_means_logp(nchans);
        {
            custom_operation op;
            op.add(smat(gpu_z2) = smat(gpu_z1) + repmat(svec(gpu_bias),1,nframes));
            op.add(svec(gpu_m2) = pow(1/cnframes*reduce_rows<add_type>(pow(smat(gpu_z2),2)),2));
            op.add(svec(gpu_m4) = 1/cnframes*reduce_rows<add_type>(pow(smat(gpu_z2),4)));
            op.add(svec(gpu_kurt) = element_div(svec(gpu_m4),svec(gpu_m2))-3);
            op.add(svec(gpu_alphas) = 5*(svec(gpu_kurt)<0) + 1*(svec(gpu_kurt)>=0));
            op.add(smat(gpu_logp) = log(repmat(svec(gpu_alphas),1,nframes))
                    - log(static_cast<NumericT>(2.0))
                    - lgamma(element_div(1,repmat(svec(gpu_alphas),1,nframes)))
                    - pow(fabs(smat(gpu_z2)),repmat(svec(gpu_alphas),1,nframes)));
            op.add(svec(gpu_means_logp) = 1/cnframes*reduce_rows<add_type>(smat(gpu_logp)));
            op.execute();
        }
        viennacl::backend::finish();
        viennacl::copy(gpu_means_logp, cmeans_logp);


        double detweights = cpu_weights.determinant();
        double H = std::log(std::abs(detweights)) + cmeans_logp.sum();

        if(grad){
            GMAT gpu_phi(nchans, nframes);
            viennacl::backend::finish();
            {
                custom_operation op;
                op.add(smat(gpu_phi) = element_prod(element_prod(repmat(svec(gpu_alphas),1,nframes), pow(fabs(smat(gpu_z2)),repmat(svec(gpu_alphas),1,nframes)-1)), sign(smat(gpu_z2))));
                op.execute();
            }
            GMAT phi_z1 = viennacl::linalg::prod(gpu_phi,viennacl::trans(gpu_z1));
            CPU_MAT cpu_phi_z1(nchans,nchans);
            viennacl::copy(phi_z1, cpu_phi_z1);
            GVEC gpu_dbias(nchans);
            viennacl::backend::finish();
            {
                custom_operation op;
                op.add(svec(gpu_dbias) = 1/cnframes*reduce_rows<add_type>(smat(gpu_phi)));
                op.execute();
            }
            viennacl::copy(gpu_dbias,cdbias);
            CPU_MAT dweights(nchans, nchans);
            dweights = (CPU_MAT::Identity(nchans,nchans) - 1/cnframes*cpu_phi_z1);
            dweights = -dweights*cpu_weights.transpose().inverse();

            std::memcpy(grad->data(), dweights.data(),sizeof(NumericT)*nchans*nchans);
            std::memcpy(grad->data()+nchans*nchans, cdbias.data(), sizeof(NumericT)*nchans);
        }

        return -H;
    }

private:
    GMAT const & data_;
};

template<class MAT>
void inplace_linear_ica(MAT & data, MAT & out){
    typedef typename MAT::value_type::value_type NumericT;
    typedef viennacl::vector<NumericT> VEC;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> CPU_MAT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic, 1> CPU_VEC;

    size_t nchans = data.size1();
    size_t nframes = data.size2();

    MAT white_data(nchans, nframes);
    whiten(data,white_data);
    CPU_VEC X = CPU_VEC::Zero(nchans*nchans + nchans);
    for(unsigned int i = 0 ; i < nchans; ++i) X[i*(nchans+1)] = 1;
    for(unsigned int i = nchans*nchans ; i < nchans*(nchans+1) ; ++i) X[i] = 0;
    ica_functor<MAT> fun(white_data);

    fmincl::optimization_options options;

    options.direction = fmincl::cg<fmincl::polak_ribiere, fmincl::no_restart>();
    options.line_search = fmincl::strong_wolfe_powell(1e-3,0.05,1.4);
    //    options.direction = fmincl::quasi_newton<fmincl::bfgs>();
    //    options.line_search = fmincl::strong_wolfe_powell(1e-4,0.9,1.4);
    options.max_iter = 2000;
    options.verbosity_level = 2;
    CPU_VEC SOL =  fmincl::minimize(fun,X, options);

    CPU_MAT cpu_optim_weights(nchans,nchans);
    CPU_VEC cpu_optim_bias(nchans);
    MAT gpu_optim_weights(nchans,nchans);
    VEC gpu_optim_bias(nchans);
    std::memcpy(cpu_optim_weights.data(), SOL.data(),sizeof(NumericT)*nchans*nchans);
    std::memcpy(cpu_optim_bias.data(), SOL.data()+nchans*nchans, sizeof(NumericT)*nchans);
    viennacl::copy(cpu_optim_weights,gpu_optim_weights);
    viennacl::copy(cpu_optim_bias,gpu_optim_bias);
    out = viennacl::linalg::prod(gpu_optim_weights,white_data);
//    {
//        custom_operation op;
//        op.add(smat(gpu_z2) = smat(gpu_z1) + repmat(svec(gpu_bias),1,nframes));
//        op.execute();
//    }
    viennacl::backend::finish();

}

template void inplace_linear_ica<viennacl::matrix<double, viennacl::row_major> >(viennacl::matrix<double, viennacl::row_major> &, viennacl::matrix<double, viennacl::row_major> &);

}

