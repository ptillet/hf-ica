/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * curveica - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <iostream>

#include "utils.hpp"

#include "curveica.h"
#include "backend.hpp"

namespace curveica{

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

    }

    template<class ScalarType>
    void compute_mean(ScalarType* A, std::size_t NC, std::size_t NF, ScalarType* x){
        for(std::size_t c = 0 ; c < NC ;++c){
            ScalarType sum = 0;
            for(std::size_t f = 0 ; f < NF ; ++f)
                sum += A[c*NF+f];
            x[c] = sum/(ScalarType)NF;
        }
    }



    template<class ScalarType>
    void whiten(std::size_t NC, std::size_t DataNF, std::size_t NF, ScalarType const * constdata, ScalarType * Sphere, ScalarType * white_data){
        ScalarType * Cov = new ScalarType[NC*NC];
        ScalarType * means = new ScalarType[NC];

        //We remove constness here to normalize the data (and add the mean back afterwards)
        ScalarType * data = const_cast<ScalarType *>(constdata);
        compute_mean(data,NC,NF,means);

        //Substract mean
        for(std::size_t c = 0 ; c < NC ;++c)
            for(std::size_t f = 0 ; f < DataNF ; ++f)
                data[c*DataNF+f] -= means[c];

        //Cov = 1/(N-1)*data_copy*data_copy'
        ScalarType alpha = (ScalarType)(1)/(NF-1);
        backend<ScalarType>::gemm(Trans,NoTrans,NC,NC,DataNF,alpha,data,DataNF,data,DataNF,0,Cov,NC);


        //Sphere = inverse(sqrtm(Cov))
        detail::inv_sqrtm<ScalarType>(NC,Cov,Sphere);

        //white_data = 2*sphere*data
        backend<ScalarType>::gemm(NoTrans,NoTrans,NF,NC,NC,2,data,DataNF,Sphere,NC,0,white_data,NF);

        //Readd mean
        for(std::size_t c = 0 ; c < NC ;++c)
            for(std::size_t f = 0 ; f < NF ; ++f)
                data[c*DataNF+f] += means[c];

        delete[] means;
        delete[] Cov;
    }

}
