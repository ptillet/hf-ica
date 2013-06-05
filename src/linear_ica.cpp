#define VIENNACL_DEBUG_BUILD

#include "clica.h"

#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"

#include "tests/benchmark-utils.hpp"
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
        t.start();
        size_t nchans = viennacl::traits::size1(data_);
        size_t nframes = viennacl::traits::size2(data_);

        NumericT cnframes = static_cast<NumericT>(nframes);

        double res;

        CPU_MAT weights(nchans, nchans);
        CPU_VEC bias(nchans), cmeans_logp(nchans);

        //Rerolls the variables into the appropriates datastructures
        std::memcpy(weights.data(), x.data(),sizeof(NumericT)*nchans*nchans);
        std::memcpy(bias.data(), x.data()+nchans*nchans, sizeof(NumericT)*nchans);

        //Creates GPU structures
        viennacl::matrix<NumericT> Gweights(nchans,nchans);
        GVEC Gbias(nchans), gm2(nchans), gm4(nchans), gkurt(nchans), galphas(nchans), means_logp(nchans), dbias(nchans);
        viennacl::copy(weights,Gweights);
        viennacl::copy(bias,Gbias);
        GMAT z1 = viennacl::linalg::prod(Gweights,data_);
        GMAT z2(nchans, nframes);
        GMAT logp(nchans, nframes);
        GMAT phi(nchans, nframes);
        {
            custom_operation op;
            op.add(smat(z2) = smat(z1) + repmat(svec(Gbias),1,nframes));
            op.execute();
        }
        viennacl::backend::finish();
        {
            custom_operation op;
            op.add(svec(gm2) = pow(1/cnframes*reduce_rows<add_type>(pow(smat(z2),2)),2));
            op.execute();
        }
        viennacl::backend::finish();
        {
            custom_operation op;
            op.add(svec(gm4) = 1/cnframes*reduce_rows<add_type>(pow(smat(z2),4)));
            op.execute();
        }
        viennacl::backend::finish();
        {
            custom_operation op;
            op.add(svec(gkurt) = element_div(svec(gm4),svec(gm2))-3);
            op.add(svec(galphas) = 1*(svec(gkurt)>0) + 4*(svec(gkurt)<=0));
            op.execute();
        }
        viennacl::backend::finish();
        {
            custom_operation op;
            op.add(smat(logp) = log(repmat(svec(galphas),1,nframes))
                    - log(static_cast<NumericT>(2.0))
                    - lgamma(element_div(1,repmat(svec(galphas),1,nframes)))
                    - pow(fabs(smat(z2)),repmat(svec(galphas),1,nframes)));
            op.execute();
        }
        viennacl::backend::finish();
        {
            custom_operation op;
            op.add(smat(phi) = element_prod(element_prod(repmat(svec(galphas),1,nframes), pow(fabs(smat(z2)),repmat(svec(galphas),1,nframes)-1)), sign(smat(z2))));
            op.execute();
        }
        viennacl::backend::finish();
        {
            custom_operation op;
            op.add(svec(means_logp) = 1/cnframes*reduce_rows<add_type>(smat(logp)));
            op.execute();
        }
        viennacl::backend::finish();
        {
            custom_operation op;
            op.add(svec(dbias) = 1/cnframes*reduce_rows<add_type>(smat(phi)));
            op.execute();
        }
        viennacl::backend::finish();
        GMAT phi_z1 = viennacl::linalg::prod(phi,viennacl::trans(z1));
        CPU_MAT invweights = weights.inverse();
        double detweights = weights.determinant();
        viennacl::copy(means_logp, cmeans_logp);
        viennacl::backend::finish();
        double H = std::log(std::abs(detweights)) + cmeans_logp.sum();
        viennacl::backend::finish();
        std::cout << galphas << std::endl;
    }

private:
    GMAT const & data_;
};

template<class MAT>
void inplace_linear_ica(MAT & data, MAT & out){
    typedef typename MAT::value_type::value_type NumericT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> CPU_MAT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic, 1> CPU_VEC;

    size_t nchans = data.size1();
    size_t nframes = data.size2();

    MAT white_data(nchans, nframes);
    whiten(data,white_data);
    CPU_VEC X = CPU_VEC::Zero(nchans*nchans + nchans);
    for(unsigned int i = 0 ; i < nchans; ++i) X[i*(nchans+1)] = 1;
    for(unsigned int i = nchans*nchans ; i < nchans*(nchans+1) ; ++i) X[i] = 2;
    ica_functor<MAT> fun(white_data);
    fun(X,NULL);

}

template void inplace_linear_ica<viennacl::matrix<double, viennacl::row_major> >(viennacl::matrix<double, viennacl::row_major> &, viennacl::matrix<double, viennacl::row_major> &);

}

