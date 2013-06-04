//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/svd.hpp"
#include "viennacl/generator/custom_operation.hpp"

namespace clica{

template<class MAT>
void whiten(MAT & data, MAT & out){
    MAT copy(data);

    //Useful typedef's
    typedef typename MAT::value_type::value_type ScalarType;
    typedef viennacl::generator::matrix<MAT> mat;
    typedef viennacl::generator::vector<ScalarType> vec;

    size_t nchans = copy.size1();
    size_t nframes = copy.size2();
    viennacl::vector<ScalarType> means(nchans);
    MAT Cov(nchans,nchans);
    ScalarType fnframes = static_cast<ScalarType>(nframes);
    {
        viennacl::generator::custom_operation op;
        op.add(vec(means) = viennacl::generator::reduce_rows<viennacl::generator::add_type>(mat(copy))/fnframes);
        op.add(mat(copy) -= viennacl::generator::repmat(vec(means),1,nframes));
        op.execute();
    }
    viennacl::backend::finish();
    Cov = viennacl::linalg::prod(copy, trans(copy));
    Cov = 1/(fnframes-1)*Cov;
    MAT Ql(nchans,nchans);
    MAT Qr(nchans,nchans);
    viennacl::vector<ScalarType> svals(nchans);
    viennacl::linalg::svd(Cov,Ql,Qr);
    Qr = viennacl::trans(Ql);
    viennacl::backend::finish();
    {
        viennacl::generator::custom_operation op;
        op.add(vec(svals) = 1/viennacl::generator::sqrt(viennacl::generator::diag(mat(Cov))));
        op.add(mat(Qr) = viennacl::generator::element_prod(mat(Qr), viennacl::generator::repmat(vec(svals),1,nframes)));
        op.execute();
    }
    viennacl::backend::finish();
    MAT Sphere = viennacl::linalg::prod(Ql,Qr);
    viennacl::backend::finish();
    Sphere = static_cast<ScalarType>(2.0)*Sphere;
    out = viennacl::linalg::prod(Sphere, data);
    viennacl::backend::finish();
}

}

template void clica::whiten<viennacl::matrix<double, viennacl::row_major> >(viennacl::matrix<double, viennacl::row_major> &, viennacl::matrix<double, viennacl::row_major> &);
