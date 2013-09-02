#ifndef PARICA_UTILS_HPP_
#define PARICA_UTILS_HPP_

#include "cblas.h"

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
    class generic_gemm;

    template<>
    class generic_gemm<double>
    {
        typedef void (*cblas_gemm_type)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const blasint, const blasint, const blasint,
                 const double, const double *, const blasint, const double *, const blasint, const double, double *, const blasint);

    public:
        static cblas_gemm_type get_ptr(){
            return &cblas_dgemm;
        }
    };

    template<>
    class generic_gemm<float>
    {
        typedef void (*cblas_gemm_type)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const blasint, const blasint, const blasint,
                 const float, const float *, const blasint, const float *, const blasint, const float, float *, const blasint);
    public:
        static cblas_gemm_type get_ptr(){
            return &cblas_sgemm;
        }
    };


}

#endif
