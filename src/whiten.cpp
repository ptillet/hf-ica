//#define VIENNACL_DEBUG_BUILD
#define VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_EIGEN

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/svd.hpp"
#include "viennacl/generator/custom_operation.hpp"


#include "Eigen/Dense"
#include "Eigen/SVD"

namespace clica{

    namespace detail{

        template<class NumericT, class F>
        static void get_sphere_gpu(viennacl::matrix<NumericT,F> & Cov, viennacl::matrix<NumericT,F> & Sphere){
            typedef viennacl::generator::matrix< viennacl::matrix<NumericT,F> > mat;
            typedef viennacl::generator::vector<NumericT> vec;

            size_t nchans = Cov.size1();
            viennacl::matrix<NumericT,F> Ql(nchans,nchans);
            viennacl::matrix<NumericT,F> Qr(nchans,nchans);
            viennacl::vector<NumericT> svals(nchans);
            viennacl::linalg::svd(Cov,Ql,Qr);
            Qr = viennacl::trans(Ql);
            viennacl::backend::finish();
            {
                viennacl::generator::custom_operation op;
                op.add(vec(svals) = 1/viennacl::generator::sqrt(viennacl::generator::diag(mat(Cov))));
                op.add(mat(Qr) = viennacl::generator::element_prod(mat(Qr), viennacl::generator::repmat(vec(svals),1,nchans)));
                op.execute();
            }
            viennacl::backend::finish();
            Sphere = viennacl::linalg::prod(Ql,Qr);
            Sphere = static_cast<NumericT>(2)*Sphere;
        }

        template<class NumericT, class F>
        static void get_sphere_cpu(viennacl::matrix<NumericT,F> & GCov, viennacl::matrix<NumericT, F> & GSphere){
            size_t nchans = GCov.size1();
            Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic>  Cov(nchans, nchans);
            Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic>  Sphere(nchans, nchans);
            viennacl::copy(GCov, Cov);
            viennacl::backend::finish();
            Eigen::JacobiSVD<Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> > svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd svals = svd.singularValues();
            for(unsigned int i = 0 ; i < nchans ; ++i) svals[i] = 1/sqrt(svals[i]);
            Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic>  U = svd.matrixU();
            Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic>  V = U.transpose();
            V = svals.asDiagonal()*V;
            Sphere = U*V;
            Sphere *= 2;
            viennacl::copy(Sphere,GSphere);
            viennacl::backend::finish();
        }

    }


    template<class MAT>
    void whiten(MAT & data, MAT & out){
        MAT copy(data);

        //Useful typedef's
        typedef typename MAT::value_type::value_type ScalarType;
        typedef viennacl::generator::matrix<MAT> mat;
        typedef viennacl::generator::vector<ScalarType> vec;

        size_t nchans = copy.size1();
        size_t nframes = copy.size2();

        MAT Sphere(nchans,nchans);
        viennacl::vector<ScalarType> means(nchans);
        MAT Cov(nchans,nchans);
        ScalarType fnframes = static_cast<ScalarType>(nframes);
        {
            viennacl::generator::custom_operation op;
            op.add(vec(means) = viennacl::generator::reduce_rows<viennacl::generator::add_type>(mat(copy))/fnframes);
            op.add(mat(copy) -= viennacl::generator::repmat(vec(means),1,nframes));
            op.execute();
            viennacl::backend::finish();
        }
        Cov = viennacl::linalg::prod(copy, trans(copy));
        Cov = 1/(fnframes-1)*Cov;
        detail::get_sphere_cpu(Cov, Sphere);
        out = viennacl::linalg::prod(Sphere, data);
        viennacl::backend::finish();
    }

}

template void clica::whiten<viennacl::matrix<double, viennacl::row_major> >(viennacl::matrix<double, viennacl::row_major> &, viennacl::matrix<double, viennacl::row_major> &);
