#include "extended_infomax.h"

#include "tests/benchmark-utils.hpp"

#include <cmath>

#include "omp.h"

#include "src/fastapprox.h"
#include <pmmintrin.h>
#include <cstddef>

namespace curveica{


template<>
void extended_infomax_ica<float>::operator()(float * z1, float * b, int const * signs, float* phi, float* means_logp) const {

#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        const __m128 bias = _mm_set1_ps(b[c]);
        int s = signs[c];
        __m128 phi_signs = _mm_set1_ps(s);
        for(unsigned int f = 0; f < NF_ ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);
            z2 = _mm_add_ps(z2,bias);

            //Computes mean_logp
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 m0_5 = _mm_set1_ps(-0.5);
            if(s<0){
                __m128 a = _mm_sub_ps(z2,_1);
                a = _mm_mul_ps(a,a);
                a = _mm_mul_ps(m0_5,a);
                a = vfastexp(a);

                __m128 b = _mm_add_ps(z2,_1);
                b = _mm_mul_ps(b,b);
                b = _mm_mul_ps(m0_5,b);
                b = vfastexp(b);

                a = _mm_add_ps(a,b);
                a = vfastlog(a);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(a));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(a,a)));
            }
            else{
                __m128 y = vfastcosh(z2);
                y = vfastlog(y);
                __m128 z2sq = _mm_mul_ps(z2,z2);
                y = _mm_mul_ps(_mm_set1_ps(-1),y);
                z2sq = _mm_mul_ps(_mm_set1_ps(0.5),z2sq);
                y = _mm_sub_ps(y,z2sq);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(y));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(y,y)));
            }

            //compute phi
            z2 = _mm_max_ps(z2,_mm_set1_ps(-10));
            z2 = _mm_min_ps(z2,_mm_set1_ps(10));
            __m128 y = vfasttanh(z2);
            y = _mm_mul_ps(phi_signs,y);
            __m128 v = _mm_add_ps(z2,y);
            _mm_store_ps(&phi[c*NF_+f],v);
        }
        double sum;
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(float)NF_*sum;
    }

}

//template<>
//void extended_infomax_ica<double>::operator()(double * z1, double * b, double * signs, double* phi, double* means_logp) const {
//    for(unsigned int c = 0 ; c < NC_ ; ++c){
//        double current = 0;
//        double k = signs[c];
//        double bias = b[c];
//        for(unsigned int f = 0; f < NF_ ; f++){
//            double z2 = z1[c*NF_+f] + bias;
//            double y = std::tanh(z2);
//            if(k<0){
//                current+= std::log((std::exp(-0.5*std::pow(z2-1,2)) + std::exp(-0.5*std::pow(z2+1,2))));
//                phi[c*NF_+f] = z2 - y;
//            }
//            else{
//                current+= -std::log(std::cosh(z2)) - 0.5*z2*z2;
//                phi[c*NF_+f] = z2 + y;
//            }
//        }
//        means_logp[c] = current/(double)NF_;
//    }
//}

template<>
void extended_infomax_ica<double>::operator()(double * z1, double * b, int const * signs, double* phi, double* means_logp) const {
#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        float k = signs[c];
        const __m128 bias = _mm_set1_ps(b[c]);
        __m128 phi_signs = (k<0)?_mm_set1_ps(-1):_mm_set1_ps(1);
        for(unsigned int f = 0; f < NF_ ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));
            z2 = _mm_add_ps(z2,bias);
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 m0_5 = _mm_set1_ps(-0.5);
            if(k<0){
                const __m128 vln0_5 = _mm_set1_ps(-0.693147);

                __m128 a = _mm_sub_ps(z2,_1);
                a = _mm_mul_ps(a,a);
                a = _mm_mul_ps(m0_5,a);
                a = vfastexp(a);

                __m128 b = _mm_add_ps(z2,_1);
                b = _mm_mul_ps(b,b);
                b = _mm_mul_ps(m0_5,b);
                b = vfastexp(b);

                a = _mm_add_ps(a,b);
                a = vfastlog(a);

                a = _mm_add_ps(vln0_5,a);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(a));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(a,a)));
            }
            else{
                __m128 y = vfastcosh(z2);
                y = vfastlog(y);
                __m128 z2sq = _mm_mul_ps(z2,z2);
                y = _mm_mul_ps(_mm_set1_ps(-1),y);
                z2sq = _mm_mul_ps(_mm_set1_ps(0.5),z2sq);
                y = _mm_sub_ps(y,z2sq);

                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(y));
                vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(y,y)));
            }

            //compute phi
            z2 = _mm_max_ps(z2,_mm_set1_ps(-10));
            z2 = _mm_min_ps(z2,_mm_set1_ps(10));
            __m128 y = vfasttanh(z2);
            y = _mm_mul_ps(phi_signs,y);
            __m128 v = _mm_add_ps(z2,y);
            _mm_store_pd(&phi[c*NF_+f],_mm_cvtps_pd(v));
            _mm_store_pd(&phi[c*NF_+f+2],_mm_cvtps_pd(_mm_movehl_ps(v,v)));
        }
        double sum;
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(double)NF_*sum;
    }
}

}
