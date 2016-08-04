/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_BACKENDS_VIENNACL_HPP
#define UMINTL_BACKENDS_VIENNACL_HPP


#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/inner_prod.hpp"

namespace umintl{

  namespace backend{

    template<class _ScalarType>
    struct viennacl_types{
        typedef _ScalarType ScalarType;
        typedef viennacl::vector<ScalarType> VectorType;
        typedef viennacl::matrix<ScalarType> MatrixType;

        static VectorType create_vector(size_t N)
        { return VectorType(N); }
        static MatrixType create_matrix(size_t M, size_t N)
        { return MatrixType(M,N); }
        static void delete_if_dynamically_allocated(VectorType const &) { }
        static void delete_if_dynamically_allocated(MatrixType const &) { }

        static void copy(size_t /*N*/, VectorType const & from, VectorType & to)
        { to = from; }
        static void axpy(size_t /*N*/, ScalarType alpha, VectorType const & x, VectorType & y)
        {  y = alpha*x + y; }
        static void scale(size_t /*N*/, ScalarType alpha, VectorType & x)
        { x = alpha*x; }
        static void scale(size_t /*M*/, size_t /*N*/, ScalarType alpha, MatrixType & A)
        { A = alpha*A; }
        static ScalarType asum(size_t /*N*/, VectorType const & x)
        { return viennacl::linalg::norm_1(x); }
        static ScalarType nrm2(size_t /*N*/, VectorType const & x)
        { return viennacl::linalg::norm_2(x); }
        static ScalarType dot(size_t /*N*/, VectorType const & x, VectorType const & y)
        { return viennacl::linalg::inner_prod(x,y); }
        static void symv(size_t /*N*/, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { y = alpha*A*x + beta*y;  }
        static void syr1(size_t /*N*/, ScalarType const & alpha, VectorType const & x, MatrixType & A)
        { A+=alpha*viennacl::linalg::outer_prod(x,x); }
        static void syr2(size_t /*N*/, ScalarType const & alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        {
          A+=alpha*viennacl::linalg::outer_prod(x,y);
          A+=alpha*viennacl::linalg::outer_prod(y,x);
        }
        static void set_to_diagonal(size_t N, MatrixType & A, ScalarType lambda)
        {
            A.resize(N,N,false);
            std::vector<ScalarType> tmp(A.internal_size1() * A.internal_size1(), 0);
            for(size_t i = 0 ; i < A.internal_size1() ; ++i)
                tmp[i*A.internal_size1()+i] = lambda;
            viennacl::fast_copy(&foo[0], &foo[0] + foo.size(), A);
        }
        static void set_to_value(VectorType & V, ScalarType val, size_t N)
        {
            V = viennacl::scalar_vector<ScalarType>(N,val);
        }
    };








  }

}

#endif
