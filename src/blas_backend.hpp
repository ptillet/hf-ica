#ifndef PARICA_OPENBLAS_BACKEND_HPP_
#define PARICA_OPENBLAS_BACKEND_HPP_

#include "lapacke.h"
#include "cblas.h"

namespace parica{

    template<class _ScalarType>
    struct blas_backend;

    template<>
    struct blas_backend<float>{
    private:
        typedef float ScalarType;
        typedef ScalarType* ptr_type;
        typedef ScalarType const * cst_ptr_type;
        typedef std::size_t size_t;
    public:
        static lapack_int getrf(int matrix_order, size_t m, size_t n, ptr_type a, size_t lda, int* ipiv)
        {    return LAPACKE_sgetrf(matrix_order,m,n,a,lda,ipiv);    }
        static lapack_int getri(int matrix_order, size_t n, ptr_type a, size_t lda, int* ipiv)
        {    return LAPACKE_sgetri(matrix_order,n,a,lda,ipiv);    }
        static void gemm(CBLAS_ORDER Order,CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K
                  , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
        {   return cblas_sgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc); }
        static lapack_int syevd( int matrix_order, char jobz, char uplo, lapack_int n,
                                   ScalarType* a, lapack_int lda, ScalarType* w )
        {   return LAPACKE_ssyevd(matrix_order,jobz,uplo,n,a,lda,w); }
    };


    template<>
    struct blas_backend<double>{
    private:
        typedef double ScalarType;
        typedef ScalarType* ptr_type;
        typedef ScalarType const * cst_ptr_type;
        typedef std::size_t size_t;
    public:
        static size_t getrf(int matrix_order, size_t m, size_t n, ptr_type a, size_t lda, int* ipiv)
        {    return LAPACKE_dgetrf(matrix_order,m,n,a,lda,ipiv);    }
        static size_t getri(int matrix_order, size_t n, ptr_type a, size_t lda, int* ipiv)
        {    return LAPACKE_dgetri(matrix_order,n,a,lda,ipiv);    }
        static void gemm(CBLAS_ORDER Order,CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K
                  , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
        {   return cblas_dgemm(Order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc); }
        static lapack_int syevd( int matrix_order, char jobz, char uplo, lapack_int n,
                                   ScalarType* a, lapack_int lda, ScalarType* w )
        {   return LAPACKE_dsyevd(matrix_order,jobz,uplo,n,a,lda,w); }
    };


}

#endif
