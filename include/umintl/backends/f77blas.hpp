/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_BACKENDS_BLAS_HPP
#define UMINTL_BACKENDS_BLAS_HPP

#include <cstring>
#include <algorithm>

namespace umintl{

  namespace backend{

    static const char Upper = 'U';
    static const char Lower = 'L';
    static const std::ptrdiff_t one_inc = 1;

    template<class _ScalarType>
    struct blas_types;

    template<>
    struct blas_types<float>{
        typedef float ScalarType;
        typedef ScalarType* VectorType;
        typedef ScalarType* MatrixType;
        typedef std::ptrdiff_t size_t;
    private:
        typedef VectorType& vec_ref;
    public:


        static VectorType create_vector(size_t N)
        { return new ScalarType[N]; }
        static MatrixType create_matrix(size_t M, size_t N)
        { return new ScalarType[M*N]; }
        static void delete_if_dynamically_allocated(ScalarType* p)
        { delete[] p;}

        static void copy(size_t N, VectorType const & from, VectorType & to)
        { FORTRAN_WRAPPER(scopy)(&N,(vec_ref)from,(size_t*)&one_inc,to,(size_t*)&one_inc); }
        static void axpy(size_t N, ScalarType alpha, VectorType const & x, VectorType & y)
        { FORTRAN_WRAPPER(saxpy)(&N,&alpha,(vec_ref)x,(size_t*)&one_inc,y,(size_t*)&one_inc); }
        static void scale(size_t N, ScalarType alpha, VectorType & x)
        { FORTRAN_WRAPPER(sscal)(&N,&alpha,x,(size_t*)&one_inc); }
        static void scale(size_t M, size_t N, ScalarType alpha, MatrixType & A)
        { size_t K=M*N; FORTRAN_WRAPPER(sscal)(&K,&alpha,A,(size_t*)&one_inc); }
        static ScalarType asum(size_t N, VectorType const & x)
        { return FORTRAN_WRAPPER(sasum)(&N,(vec_ref)x,(size_t*)&one_inc);}
        static ScalarType nrm2(size_t N, VectorType const & x)
        { return FORTRAN_WRAPPER(snrm2)(&N,(vec_ref)x,(size_t*)&one_inc); }
        static ScalarType dot(size_t N, VectorType const & x, VectorType const & y)
        { return FORTRAN_WRAPPER(sdot)(&N,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc); }
        static void symv(size_t N, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { FORTRAN_WRAPPER(ssymv)((char*)&Lower,&N,&alpha,A,&N,(vec_ref)x,(size_t*)&one_inc,&beta,y,(size_t*)&one_inc);  }
        static void syr1(size_t N, ScalarType alpha, VectorType const & x, MatrixType & A)
        { FORTRAN_WRAPPER(ssyr)((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,A,&N); }
        static void syr2(size_t N, ScalarType  alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        { FORTRAN_WRAPPER(ssyr2)((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc,A,&N); }
        static void set_to_value(VectorType & V, ScalarType val, size_t N)
        { std::fill(V, V+N, val); }
        static void set_to_diagonal(size_t N, MatrixType & A, ScalarType lambda) {
            std::memset(A,0,N*N*sizeof(ScalarType));
            for(size_t i = 0 ; i < N ; ++i){
                A[i*N+i] = lambda;
            }
        }
    };


    template<>
    struct blas_types<double>{
        typedef double ScalarType;
        typedef ScalarType* VectorType;
        typedef ScalarType* MatrixType;
        typedef std::ptrdiff_t size_t;

    private:
        typedef VectorType& vec_ref;
    public:


        static VectorType create_vector(size_t N)
        { return new ScalarType[N]; }
        static MatrixType create_matrix(size_t M, size_t N)
        { return new ScalarType[M*N]; }
        static void delete_if_dynamically_allocated(ScalarType* p)
        { delete[] p;}

        static void copy(size_t N, VectorType const & from, VectorType & to)
        { FORTRAN_WRAPPER(dcopy)(&N,(vec_ref)from,(size_t*)&one_inc,to,(size_t*)&one_inc); }
        static void axpy(size_t N, ScalarType alpha, VectorType const & x, VectorType & y)
        { FORTRAN_WRAPPER(daxpy)(&N,&alpha,(vec_ref)x,(size_t*)&one_inc,y,(size_t*)&one_inc); }
        static void scale(size_t N, ScalarType alpha, VectorType & x)
        { FORTRAN_WRAPPER(dscal)(&N,&alpha,x,(size_t*)&one_inc); }
        static void scale(size_t M, size_t N, ScalarType alpha, MatrixType & A)
        { size_t K=M*N; FORTRAN_WRAPPER(dscal)(&K,&alpha,A,(size_t*)&one_inc); }
        static ScalarType asum(size_t N, VectorType const & x)
        { return FORTRAN_WRAPPER(dasum)(&N,(vec_ref)x,(size_t*)&one_inc);}
        static ScalarType nrm2(size_t N, VectorType const & x)
        { return FORTRAN_WRAPPER(dnrm2)(&N,(vec_ref)x,(size_t*)&one_inc); }
        static ScalarType dot(size_t N, VectorType const & x, VectorType const & y)
        { return FORTRAN_WRAPPER(ddot)(&N,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc); }
        static void symv(size_t N, ScalarType alpha, MatrixType const& A, VectorType const & x, ScalarType beta, VectorType & y)
        { FORTRAN_WRAPPER(dsymv)((char*)&Lower,&N,&alpha,A,&N,(vec_ref)x,(size_t*)&one_inc,&beta,y,(size_t*)&one_inc);  }
        static void syr1(size_t N, ScalarType alpha, VectorType const & x, MatrixType & A)
        { FORTRAN_WRAPPER(dsyr)((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,A,&N); }
        static void syr2(size_t N, ScalarType  alpha, VectorType const & x, VectorType const & y, MatrixType & A)
        { FORTRAN_WRAPPER(dsyr2)((char*)&Lower,&N,&alpha,(vec_ref)x,(size_t*)&one_inc,(vec_ref)y,(size_t*)&one_inc,A,&N); }
        static void set_to_value(VectorType & V, ScalarType val, size_t N)
        { std::fill(V, V+N, val); }
        static void set_to_diagonal(size_t N, MatrixType & A, ScalarType lambda) {
            std::memset(A,0,N*N*sizeof(ScalarType));
            for(size_t i = 0 ; i < N ; ++i){
                A[i*N+i] = lambda;
            }
        }
    };


  }

}

#endif
