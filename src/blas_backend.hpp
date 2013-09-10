#ifndef PARICA_OPENBLAS_BACKEND_HPP_
#define PARICA_OPENBLAS_BACKEND_HPP_

#include "lapacke.h"
#include "cblas.h"
#include "fmincl/backends/openblas.hpp"

namespace parica{

    template<class ScalarType>
    struct fmincl_backend{
        typedef typename fmincl::backend::OpenBlasTypes<ScalarType> type;
    };

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
        static lapack_int getrf(size_t m, size_t n, ptr_type a, size_t lda, int* ipiv)
        {   return LAPACKE_sgetrf(LAPACK_COL_MAJOR,m,n,a,lda,ipiv);    }
        static lapack_int getri(size_t n, ptr_type a, size_t lda, int* ipiv)
        {   return LAPACKE_sgetri(LAPACK_COL_MAJOR,n,a,lda,ipiv);    }
        static void gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
        {   return cblas_sgemm(CblasColMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc); }
        static lapack_int syevd(char jobz, char uplo, lapack_int n,  ScalarType* a, lapack_int lda, ScalarType* w )
        {   return LAPACKE_ssyevd(CblasColMajor,jobz,uplo,n,a,lda,w); }
    };


    template<>
    struct blas_backend<double>{
    private:
        typedef double ScalarType;
        typedef ScalarType* ptr_type;
        typedef ScalarType const * cst_ptr_type;
        typedef std::size_t size_t;
    public:
        static size_t getrf(size_t m, size_t n, ptr_type a, size_t lda, int* ipiv)
        {    return LAPACKE_dgetrf(LAPACK_COL_MAJOR,m,n,a,lda,ipiv);    }
        static size_t getri(size_t n, ptr_type a, size_t lda, int* ipiv)
        {    return LAPACKE_dgetri(LAPACK_COL_MAJOR,n,a,lda,ipiv);    }
        static void gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
        {   return cblas_dgemm(CblasColMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc); }
        static lapack_int syevd( char jobz, char uplo, lapack_int n, ScalarType* a, lapack_int lda, ScalarType* w )
        {   return LAPACKE_dsyevd(LAPACK_COL_MAJOR,jobz,uplo,n,a,lda,w); }
    };


}

#endif
