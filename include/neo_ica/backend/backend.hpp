/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef NEO_ICA_BACKEND_HPP_
#define NEO_ICA_BACKEND_HPP_

#include <stdlib.h>
#include "blas.h"
#include "lapack.h"
#include "umintl/backends/f77blas.hpp"


namespace neo_ica{

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
    {   sgetrf(&m,&n,a,&lda,(size_t*)ipiv,&dummy_info);    }
    static void getri(size_t n, ptr_type A, size_t lda, size_t* ipiv)
    {
        size_t lwork = -1;
        ScalarType* work = new ScalarType;
        sgetri(&n, A, &lda, ipiv, work, &lwork, &dummy_info);
        lwork = (size_t) work[0];
        delete work;
        work = new ScalarType[lwork];
        sgetri(&n, A, &lda, ipiv, work, &lwork, &dummy_info);
        delete[] work;
    }
    static void gemm(char TransA, char TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
    {   sgemm(&TransA,&TransB,&M,&N,&K,&alpha,(ptr_type)A,&lda,(ptr_type)B,&ldb,&beta,C,&ldc); }
    static void syev(char jobz, char uplo, size_t n,  ScalarType* a, size_t lda, ScalarType* w )
    {
        size_t lwork = -1;
        ScalarType* work = new ScalarType;
        ssyev(&jobz,&uplo, &n, a, &lda, w, work, &lwork, &dummy_info );
        lwork = (size_t) work[0];
        delete work;
        work = new ScalarType[lwork];
        ssyev(&jobz,&uplo,&n,a,&lda,w,work,&lwork,&dummy_info);
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
    {   dgetrf(&m,&n,a,&lda,(size_t*)ipiv,&dummy_info);    }
    static void getri(size_t n, ptr_type A, size_t lda, size_t* ipiv)
    {
        size_t lwork = -1;
        ScalarType* work = (ScalarType *) malloc(sizeof(ScalarType) * 1);
        dgetri(&n, A, &lda, ipiv, work, &lwork, &dummy_info);
        lwork = (size_t) work[0];
        free(work);
        work = (ScalarType *) malloc(sizeof(ScalarType) * lwork);
        dgetri(&n, A, &lda, ipiv, work, &lwork, &dummy_info);
        free(work);
    }
    static void gemm(char TransA, char TransB, size_t M, size_t N, size_t K , ScalarType alpha, cst_ptr_type A, size_t lda, cst_ptr_type B, size_t ldb, ScalarType beta, ptr_type C, size_t ldc)
    {   dgemm(&TransA,&TransB,&M,&N,&K,&alpha,(ptr_type)A,&lda,(ptr_type)B,&ldb,&beta,C,&ldc); }
    static void syev(char jobz, char uplo, size_t n,  ScalarType* a, size_t lda, ScalarType* w )
    {
        size_t lwork = -1;
        ScalarType* work = new ScalarType;
        dsyev(&jobz,&uplo, &n, a, &lda, w, work, &lwork, &dummy_info );
        lwork = (size_t) work[0];
        delete work;
        work = new ScalarType[lwork];
        dsyev(&jobz,&uplo,&n,a,&lda,w,work,&lwork,&dummy_info);
        delete[] work;
    }
};

}

#endif
