/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * DSHF-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef dshf_ica_BACKEND_HPP_
#define dshf_ica_BACKEND_HPP_

#ifdef dshf_ica_WITH_BLAS

using std::ptrdiff_t;

#include "lapack.h"
#include "blas.h"
#include "umintl/backends/blas.hpp"

#else

#ifdef __cplusplus
extern "C"{
#endif

#include "lapacke.h"
#include "cblas.h"

#ifdef __cplusplus
}
#endif

#include "umintl/backends/cblas.hpp"


#endif

namespace dshf_ica{


#ifdef dshf_ica_WITH_BLAS

static std::ptrdiff_t dummy_info;

static const char Trans = 'T';
static const char NoTrans = 'N';

template<class ScalarType>
struct umintl_backend{
    typedef typename umintl::backend::blas_types<ScalarType> type;
};


template<class _ScalarType>
struct backend;

template<>
struct backend<float>{
    typedef float ScalarType;
    typedef ScalarType* ptr_type;
    typedef ScalarType const * cst_ptr_type;
    typedef std::ptrdiff_t size_t;


    static void getrf(size_t m, size_t n, ptr_type a, size_t lda, size_t* ipiv)
    {   sgetrf_(&m,&n,a,&lda,(size_t*)ipiv,&dummy_info);    }
    static void getri(size_t n, ptr_type A, size_t lda, size_t* ipiv)
    {
        size_t lwork = -1;
        ScalarType* work = new ScalarType;
        sgetri_(&n, A, &n, ipiv, work, &lwork, &dummy_info);
        lwork = (size_t) work[0];
        delete work;
        work = new ScalarType[lwork];
        sgetri_(&n, A, &n, ipiv, work, &lwork, &dummy_info);
        delete[] work;
    }
    static void gemm(char TransA, char TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
    {   sgemm_(&TransA,&TransB,&M,&N,&K,&alpha,(ptr_type)A,&lda,(ptr_type)B,&ldb,&beta,C,&ldc); }
    static void syev(char jobz, char uplo, size_t n,  ScalarType* a, size_t lda, ScalarType* w )
    {
        size_t lwork = -1;
        ScalarType* work = new ScalarType;
        ssyev_(&jobz,&uplo, &n, a, &lda, w, work, &lwork, &dummy_info );
        lwork = (size_t) work[0];
        delete work;
        work = new ScalarType[lwork];
        ssyev_(&jobz,&uplo,&n,a,&lda,w,work,&lwork,&dummy_info);
        delete[] work;
    }
};


template<>
struct backend<double>{
    typedef double ScalarType;
    typedef ScalarType* ptr_type;
    typedef ScalarType const * cst_ptr_type;
    typedef std::ptrdiff_t size_t;

    static void getrf(size_t m, size_t n, ptr_type a, size_t lda, size_t* ipiv)
    {   dgetrf_(&m,&n,a,&lda,(size_t*)ipiv,&dummy_info);    }
    static void getri(size_t n, ptr_type A, size_t lda, size_t* ipiv)
    {
        size_t lwork = -1;
        ScalarType* work = (ScalarType *) malloc(sizeof(ScalarType) * 1);
        dgetri_(&n, A, &n, ipiv, work, &lwork, &dummy_info);
        lwork = (size_t) work[0];
        free(work);
        work = (ScalarType *) malloc(sizeof(ScalarType) * lwork);
        dgetri_(&n, A, &n, ipiv, work, &lwork, &dummy_info);
    }
    static void gemm(char TransA, char TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
    {   dgemm_(&TransA,&TransB,&M,&N,&K,&alpha,(ptr_type)A,&lda,(ptr_type)B,&ldb,&beta,C,&ldc); }
    static void syev(char jobz, char uplo, size_t n,  ScalarType* a, size_t lda, ScalarType* w )
    {
        size_t lwork = -1;
        ScalarType* work = new ScalarType;
        dsyev_(&jobz,&uplo, &n, a, &lda, w, work, &lwork, &dummy_info );
        lwork = (size_t) work[0];
        delete work;
        work = new ScalarType[lwork];
        dsyev_(&jobz,&uplo,&n,a,&lda,w,work,&lwork,&dummy_info);
        delete[] work;
    }
};


#else
static const CBLAS_TRANSPOSE Trans = CblasTrans;
static const CBLAS_TRANSPOSE NoTrans = CblasNoTrans;

template<class ScalarType>
struct umintl_backend{
    typedef typename umintl::backend::cblas_types<ScalarType> type;
};


template<class _ScalarType>
struct backend;

template<>
struct backend<float>{
    typedef float ScalarType;
    typedef ScalarType* ptr_type;
    typedef ScalarType const * cst_ptr_type;
    typedef int size_t;

    static lapack_int getrf(size_t m, size_t n, ptr_type a, size_t lda, int* ipiv)
    {   return LAPACKE_sgetrf(LAPACK_COL_MAJOR,m,n,a,lda,ipiv);    }
    static lapack_int getri(size_t n, ptr_type a, size_t lda, int* ipiv)
    {   return LAPACKE_sgetri(LAPACK_COL_MAJOR,n,a,lda,ipiv);    }
    static void gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
    {   return cblas_sgemm(CblasColMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc); }
    static lapack_int syev(char jobz, char uplo, lapack_int n,  ScalarType* a, lapack_int lda, ScalarType* w )
    {   return LAPACKE_ssyev(CblasColMajor,jobz,uplo,n,a,lda,w); }
    static void axpy(std::size_t N, ScalarType alpha, cst_ptr_type x, ptr_type y)
    { cblas_saxpy(N,alpha,x,1,y,1); }
};


template<>
struct backend<double>{
    typedef double ScalarType;
    typedef ScalarType* ptr_type;
    typedef ScalarType const * cst_ptr_type;
    typedef int size_t;

    static size_t getrf(size_t m, size_t n, ptr_type a, size_t lda, int* ipiv)
    {    return LAPACKE_dgetrf(LAPACK_COL_MAJOR,m,n,a,lda,ipiv);    }
    static size_t getri(size_t n, ptr_type a, size_t lda, int* ipiv)
    {    return LAPACKE_dgetri(LAPACK_COL_MAJOR,n,a,lda,ipiv);    }
    static void gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
    {   return cblas_dgemm(CblasColMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc); }
    static lapack_int syev( char jobz, char uplo, lapack_int n, ScalarType* a, lapack_int lda, ScalarType* w )
    {   return LAPACKE_dsyev(LAPACK_COL_MAJOR,jobz,uplo,n,a,lda,w); }
    static void axpy(std::size_t N, ScalarType alpha, cst_ptr_type x, ptr_type y)
    { cblas_daxpy(N,alpha,x,1,y,1); }
};


#endif

}

#endif
