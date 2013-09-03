#ifndef PARICA_UTILS_HPP_
#define PARICA_UTILS_HPP_

#include "cblas.h"
#include "lapacke.h"
#include "Eigen/Dense"

namespace parica{

    template<int N>
    struct compile_time_pow{
        template<class ScalarType>
        ScalarType operator()(ScalarType v){
            return v*compile_time_pow<N-1>()(v);
        }
    };

    template<>
    struct compile_time_pow<0>{
        template<class ScalarType>
        ScalarType operator()(ScalarType v){
            return 1;
        }
    };




    template<class ScalarType>
    class generic_getrf;

    template<>
    struct generic_getrf<double>{
        typedef lapack_int (*fun_type)(int matrix_order, lapack_int m, lapack_int n,
                                           double* a, lapack_int lda, lapack_int* ipiv );
        static fun_type get_ptr(){  return &LAPACKE_dgetrf; }
    };

    template<>
    struct generic_getrf<float>{
        typedef lapack_int (*fun_type)(int matrix_order, lapack_int m, lapack_int n,
                                           float* a, lapack_int lda, lapack_int* ipiv );
        static fun_type get_ptr(){ return &LAPACKE_sgetrf; }
    };


    template<class ScalarType>
    class generic_getri;

    template<>
    struct generic_getri<double>{
        typedef lapack_int (*fun_type)( int matrix_order, lapack_int n, double* a,
                                        lapack_int lda, const lapack_int* ipiv );
        static fun_type get_ptr(){  return &LAPACKE_dgetri; }
    };

    template<>
    struct generic_getri<float>{
        typedef lapack_int (*fun_type)( int matrix_order, lapack_int n, float* a,
                                        lapack_int lda, const lapack_int* ipiv );
        static fun_type get_ptr(){ return &LAPACKE_sgetri; }
    };


    template<class ScalarType>
    class generic_gemm;

    template<>
    struct generic_gemm<double>{
        typedef void (*fun_type)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const blasint, const blasint, const blasint
                                        ,const double, const double *, const blasint, const double *, const blasint, const double, double *, const blasint);
        static fun_type get_ptr(){  return &cblas_dgemm; }
    };

    template<>
    struct generic_gemm<float>{
        typedef void (*fun_type)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const blasint, const blasint, const blasint,const float, const float *, const blasint, const float *, const blasint
                                        , const float, float *, const blasint);
        static fun_type get_ptr(){ return &cblas_sgemm; }
    };

    template<class MatrixType>
    static void inplace_inverse(MatrixType & A)
    {
        assert(A.rows()==A.cols() && "Input is not square");
        lapack_int order = MatrixType::IsRowMajor?LAPACK_ROW_MAJOR:LAPACK_COL_MAJOR;
        std::size_t N = A.rows();
        int *IPIV = new int[N+1];
        int INFO;

        INFO = (*generic_getrf<typename MatrixType::Scalar>::get_ptr())(order,N,N,A.data(),N,IPIV);
        INFO = (*generic_getri<typename MatrixType::Scalar>::get_ptr())(order,N,A.data(),N,IPIV);

        delete IPIV;
    }

    template<class MatrixType>
    static void gemm(typename MatrixType::Scalar alpha, MatrixType const & A, MatrixType const & B, typename MatrixType::Scalar beta, MatrixType & C){
        assert(A.rows()==C.rows() && B.cols()==C.cols() && A.cols()==B.cols()() && "Incompatible sizes");
        int lda = MatrixType::IsRowMajor?A.cols():A.rows();
        int ldb = MatrixType::IsRowMajor?B.cols():B.rows();
        int ldc = MatrixType::IsRowMajor?C.cols():C.rows();
        (*generic_gemm<typename MatrixType::Scalar>::get_ptr())(MatrixType::IsRowMajor?CblasRowMajor:CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(), A.cols()
                                                                ,alpha,A.data(),lda,B.data(),ldb,beta,C.data(),ldc);
    }

    template<class MatrixType>
    static void gemm(typename MatrixType::Scalar alpha, MatrixType const & A, Eigen::Transpose<MatrixType> const & B, typename MatrixType::Scalar beta, MatrixType & C){
        assert(A.rows()==C.rows() && B.cols()==C.cols() && A.cols()==B.cols()() && "Incompatible sizes");
        int lda = MatrixType::IsRowMajor?A.cols():A.rows();
        int ldb = MatrixType::IsRowMajor?B.rows():B.cols();
        int ldc = MatrixType::IsRowMajor?C.cols():C.rows();
        (*generic_gemm<typename MatrixType::Scalar>::get_ptr())(MatrixType::IsRowMajor?CblasRowMajor:CblasColMajor, CblasNoTrans, CblasTrans, A.rows(), B.cols(), A.cols()
                                                                ,alpha,A.data(),lda,B.data(),ldb,beta,C.data(),ldc);
    }

}

#endif
