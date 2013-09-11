/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <iostream>

#include "utils.hpp"

#include "parica.h"
#include "backend.hpp"

namespace parica{

    namespace detail{

        template<class ScalarType>
        static void inv_sqrtm(std::size_t C, ScalarType * in, ScalarType * out){

            ScalarType * D = new ScalarType[C];
            ScalarType * UD = new ScalarType[C*C];

            //in = U
            backend<ScalarType>::syev('V','U',C,in,C,D);
            //UD = U*diag(D)
            for (std::size_t j=0; j<C; ++j) {
              ScalarType lambda = 1/std::sqrt(D[j]);
              for (std::size_t i=0; i<C; ++i)
                  UD[j*C+i] = in[j*C+i]*lambda;
            }

            //out = UD*U^T
            backend<ScalarType>::gemm(NoTrans,Trans,C,C,C,1,UD,C,in,C,0,out,C);

            delete[] D;
            delete[] UD;
        }

        template<class ScalarType>
        void mean(ScalarType* A, std::size_t NC, std::size_t NF, ScalarType* x){
            for(std::size_t c = 0 ; c < NC ;++c){
                ScalarType sum = 0;
                for(std::size_t f = 0 ; f < NF ; ++f)
                    sum += A[c*NF+f];
                x[c] = sum/(ScalarType)NF;
            }
        }


        template<class ScalarType>
        void normalize(ScalarType* A, std::size_t NC, std::size_t NF){
            ScalarType * x = new ScalarType[NC];
            mean(A,NC,NF,x);
            for(std::size_t c = 0 ; c < NC ;++c)
                for(std::size_t f = 0 ; f < NF ; ++f)
                    A[c*NF+f] -= x[c];
            delete[] x;
        }

    }


    template<class ScalarType>
    void whiten(std::size_t nchans, std::size_t nframes, ScalarType * data, ScalarType * out){
        ScalarType * Cov = new ScalarType[nchans*nchans];
        ScalarType * Sphere = new ScalarType[nchans*nchans];

        //data_copy -= mean(data_copy,2);
        detail::normalize(data,nchans,nframes);

        //Cov = 1/(N-1)*data_copy*data_copy'
        ScalarType alpha = (ScalarType)(1)/(nframes-1);
        backend<ScalarType>::gemm(Trans,NoTrans,nchans,nchans,nframes,alpha,data,nframes,data,nframes,0,Cov,nchans);

        //Sphere = inverse(sqrtm(Cov))
        detail::inv_sqrtm<ScalarType>(nchans,Cov,Sphere);

        //out = 2*Sphere*data_copy;
        backend<ScalarType>::gemm(NoTrans,Trans,nframes,nchans,nchans,2,data,nframes,Sphere,nchans,0,out,nframes);

        delete[] Cov;
        delete[] Sphere;
    }

}
