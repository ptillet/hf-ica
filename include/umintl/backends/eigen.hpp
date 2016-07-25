/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_BACKENDS_EIGEN_HPP
#define UMINTL_BACKENDS_EIGEN_HPP


#include "Eigen/Dense"

namespace umintl{

  namespace backend{

    template<class _ScalarType>
    struct eigen_types{
        typedef _ScalarType ScalarType;
        typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorType;
        typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

        static VectorType create_vector(std::size_t N)
        { return VectorType(N); }
        static MatrixType create_matrix(std::size_t M, std::size_t N)
        { return MatrixType(M,N); }
        static void delete_if_dynamically_allocated(VectorType const &) { }
        static void delete_if_dynamically_allocated(MatrixType const &) { }

        static void copy(std::size_t /*N*/, VectorType const & from, VectorType & to)
        { to = from; }
        static void axpy(std::size_t /*N*/, ScalarType alpha, VectorType const & x, VectorType & y)
        {  y = alpha*x + y; }
        static void scale(std::size_t /*N*/, ScalarType alpha, VectorType & x)
        { x = alpha*x; }
        static void scale(std::size_t /*M*/, std::size_t /*N*/, ScalarType alpha, MatrixType & A)
        { A = alpha*A; }
        static ScalarType asum(std::size_t /*N*/, VectorType const & x)
        { return x.array().abs().sum(); }
        static ScalarType nrm2(std::size_t /*N*/, VectorType const & x)
        { return x.norm(); }
        static ScalarType dot(std::size_t /*N*/, VectorType const & x, VectorType const & y)
        { return x.dot(y); }
        static void symv(std::size_t /*N*/, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { y = alpha*A*x + beta*y;  }
        static void gemv(std::size_t /*M*/, std::size_t /*N*/, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { y = alpha*A*x + beta*y;  }
        static void syr1(std::size_t /*N*/, ScalarType const & alpha, VectorType const & x, MatrixType & A)
        { A+=alpha*x*x.transpose(); }
        static void syr2(std::size_t /*N*/, ScalarType const & alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        { A+=alpha*x*y.transpose() + alpha*y*x.transpose(); }
        static void set_to_diagonal(std::size_t N, MatrixType & A, ScalarType lambda)
        { A = lambda*MatrixType::Identity(N, N); }
        static void set_to_value(VectorType & V, ScalarType val, std::size_t N)
        { V = VectorType::Constant(N,val); }
    };








  }

}

#endif
