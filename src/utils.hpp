#ifndef PARICA_UTILS_HPP_
#define PARICA_UTILS_HPP_

#include "openblas_backend.hpp"
#include "Eigen/Dense"

namespace parica{

    namespace detail{

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
        static void inplace_inverse(lapack_int order, std::size_t N, ScalarType * A)
        {
            int *ipiv = new int[N+1];
            int info;
            info = openblas_backend<ScalarType>::getrf(order,N,N,A,N,ipiv);
            info = openblas_backend<ScalarType>::getri(order,N,A,N,ipiv);
            delete ipiv;
        }

    }

}

#endif
