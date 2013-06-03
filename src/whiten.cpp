#define VIENNACL_DEBUG_BUILD

#include "viennacl/matrix.hpp"
#include "viennacl/generator/custom_operation.hpp"

//  normX = X - repmat(mean(X,2),1,size(X,2));
//	C = 1/(size(X,2)-1)*normX*normX';
//	[U D Ut] = svd(C);
//	sphere = 2.0*U*diag(1./sqrt(diag(D)))*U';
//	M = sphere*normX;

//typedef typename MatT::value_type::value_type ScalarType;
//typedef generator::dummy_matrix<MatT> dm;
//typedef generator::dummy_vector<ScalarType> dv;
//viennacl::generator::custom_operation op, op2;

//viennacl::generator::custom_operation op;
//viennacl::vector<ScalarType> means(mat.size1());
//op.add(dv(means) = generator::reduce_cols<generator::add_type>(dm(mat))/(float)mat.size1());
//op.add(dm(mat) -= generator::trans(repmat(dv(means),1,mat.size1())));
//op.execute();
//ocl::get_queue().finish();

//MatT copy(mat);
//MatT Ql(mat.size1(),mat.size1());
//MatT Qr(mat.size2(),mat.size2());
//viennacl::vector<ScalarType> svals(mat.size1());
//viennacl::linalg::svd(mat,Ql,Qr);
//viennacl::ocl::get_queue().finish();
//op.add(dv(svals) = generator::diag(dm(mat)));
//op.add(dm(mat) = generator::prod(dm(copy),dm(Qr)));
//op.add(dm(mat) = generator::element_div(dm(mat),epsilon + generator::trans(generator::repmat(dv(svals),1,mat.size1()))));
//op.execute();
//viennacl::ocl::get_queue().finish();

namespace clica{

template<class MAT>
void whiten(MAT & data, MAT & out){
    //Useful typedef's
    typedef typename MAT::value_type::value_type ScalarType;
    typedef viennacl::generator::matrix<MAT> mat;
    typedef viennacl::generator::vector<ScalarType> vec;

    viennacl::vector<ScalarType> means;
    MAT Cov;
    size_t nchans = data.size1();
    size_t nframes = data.size2();
    ScalarType cnframes = static_cast<ScalarType>(nframes);


    {
        viennacl::generator::custom_operation op;
        op.add(vec(means) = viennacl::generator::reduce_rows<viennacl::generator::add_type>(mat(data))/cnframes);
        op.add(mat(data) -= viennacl::generator::repmat(vec(means),1,nframes));
//        op.add(mat(Cov) = 1/(cnframes-1)*viennacl::generator::prod(mat(data),viennacl::generator::trans(mat(data))));
        op.execute();
        viennacl::backend::finish();
    }

}

}

template void clica::whiten<viennacl::matrix<double, viennacl::row_major> >(viennacl::matrix<double, viennacl::row_major> &, viennacl::matrix<double, viennacl::row_major> &);
