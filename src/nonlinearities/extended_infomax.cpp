/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cmath>
#include <cstddef>
#include "omp.h"

#include "neo_ica/nonlinearities/extended_infomax.h"

#include "tests/benchmark-utils.hpp"
#include "src/math/math.h"
#include "../tools/round.hpp"


namespace neo_ica{

using namespace std;
using namespace math;

//template<>
//void extended_infomax_ica<float>::compute_phi(size_t offset, size_t sample_size, float * z1, int const * signs, float* phi) const{
//    for(size_t c = 0 ; c < NC_ ; ++c){
//        float k = (signs[c]>0)?1:-1;
//        for(size_t f = offset ; f < offset + sample_size ; ++f)
//          phi[c*NF_+f] = z1[c*NF_+f] + k*tanh(z1[c*NF_+f]);
//    }
//}

//template<>
//void extended_infomax_ica<float>::compute_dphi(size_t offset, size_t sample_size, float * z1, int const * signs, float* dphi) const {
//    for(size_t c = 0 ; c < NC_ ; ++c){
//        float s = signs[c];
//        for(size_t f = offset ; f < offset + sample_size ; ++f){
//          float y = tanh(z1[c*NF_+f]);
//          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
//        }
//    }
//}

//template<>
//void extended_infomax_ica<float>::compute_means_logp(size_t offset, size_t sample_size, float * z1, int const * signs, float* means_logp) const {
//    for(size_t c = 0 ; c < NC_ ; ++c){
//        float sum = 0;
//        float k = signs[c];
//        for(size_t f = offset ; f < offset + sample_size ; ++f){
//          float z = z1[c*NF_+f];
//          sum+=(k<0)? - 0.693147 - 0.5*(z-1)*(z-1) + log(1+exp(-2*z)):-log(cosh(z))-0.5*z*z;
//        }
//        means_logp[c] = 1/(float)sample_size*sum;
//    }
//}



//template<>
//void extended_infomax_ica<double>::compute_phi(size_t offset, size_t sample_size, double * z1, int const * signs, double* phi) const {
//    for(size_t c = 0 ; c < NC_ ; ++c){
//        double k = (signs[c]>0)?1:-1;
//        for(size_t f = offset ; f < offset + sample_size ; ++f)
//          phi[c*NF_+f] = z1[c*NF_+f] + k*tanh(z1[c*NF_+f]);
//    }
//}

//template<>
//void extended_infomax_ica<double>::compute_dphi(size_t offset, size_t sample_size, double * z1, int const * signs, double* dphi) const {
//    for(size_t c = 0 ; c < NC_ ; ++c){
//        double s = signs[c];
//        for(size_t f = offset ; f < offset + sample_size ; ++f){
//          double y = tanh(z1[c*NF_+f]);
//          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
//        }
//    }
//}

//template<>
//void extended_infomax_ica<double>::compute_means_logp(size_t offset, size_t sample_size, double * z1, int const * signs, double* means_logp) const {
//    for(size_t c = 0 ; c < NC_ ; ++c){
//        double sum = 0;
//        double k = signs[c];
//        for(size_t f = offset ; f < offset + sample_size ; ++f){
//          double z = z1[c*NF_+f];
//          sum+=(k<0)? - 0.693147 - 0.5*(z-1)*(z-1) + log(1+exp(-2*z)):-log(cosh(z))-0.5*z*z;
//        }
//        means_logp[c] = 1/(double)sample_size*sum;
//    }
//}

template<>
void extended_infomax_ica<float>::compute_phi(size_t offset, size_t sample_size, float * z1, int const * signs, float* phi) const {
#pragma omp parallel for
    for(size_t c = 0 ; c < NC_ ; ++c){
        int s = signs[c];
        __m128 phi_signs = _mm_set1_ps(s);
        size_t f = offset;
        for(; f < tools::round_to_next_multiple(offset,4); ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + s*tanh(z1[c*NF_+f]);
        for(; f < tools::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);
            //compute phi
            __m128 y = vtanh(z2);
            y = _mm_mul_ps(phi_signs,y);
            __m128 v = _mm_add_ps(z2,y);
            _mm_store_ps(&phi[c*NF_+f],v);
        }
        for(; f < offset+sample_size ; ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + s*tanh(z1[c*NF_+f]);
    }

}

template<>
void extended_infomax_ica<float>::compute_dphi(size_t offset, size_t sample_size, float * z1, int const * signs, float* dphi) const {
    #pragma omp parallel for
    for(size_t c = 0 ; c < NC_ ; ++c){
        int s = signs[c];
        size_t f = offset;
        for(; f < tools::round_to_next_multiple(offset,4); ++f){
          float y = tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
        for(; f < tools::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);
            __m128 y = vtanh(z2);
            __m128 val;
            if(s>0)
                val =  _mm_set1_ps(2) - y*y;
            else
                val = y*y;
            _mm_store_ps(&dphi[c*NF_+f],val);
        }
        for(; f < offset+sample_size ; ++f){
          float y = tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
    }
}

template<>
void extended_infomax_ica<float>::compute_means_logp(size_t offset, size_t sample_size, float * z1, int const * signs, float* means_logp) const {
    #pragma omp parallel for
    for(size_t c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        int s = signs[c];
        double sum = 0;
        size_t f = offset;
        for(; f < tools::round_to_next_multiple(offset,4); ++f){
          float z = z1[c*NF_+f];
          sum+=(s<0)? - 0.693147 - 0.5*(z-1)*(z-1) + log(1+exp(-2*z)):-log(cosh(z))-0.5*z*z;
        }
        for(; f < tools::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);

            //Computes mean_logp
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 _0_5 = _mm_set1_ps(0.5);
            const __m128 _2 = _mm_set1_ps(2);
            __m128 val;
            if(s<0)
                val = _mm_set1_ps(-0.693147) - _0_5*(z2 - _1)*(z2 - _1) + vlog(_1 + vexp(-_2*z2));
            else
                val = - vlog(vcosh(z2)) - _0_5*z2*z2;
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(val));
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(val,val)));
        }
        for(; f < offset+sample_size; ++f){
          float z = z1[c*NF_+f];
          sum+=(s<0)? - 0.693147 - 0.5*(z-1)*(z-1) + log(1+exp(-2*z)):-log(cosh(z))-0.5*z*z;
        }
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(double)sample_size*sum;
    }

}

template<>
void extended_infomax_ica<double>::compute_phi(size_t offset, size_t sample_size, double * z1, int const * signs, double* phi) const {
    #pragma omp parallel for
    for(size_t c = 0 ; c < NC_ ; ++c){
        float k = (signs[c]>0)?1:-1;
        __m128 phi_signs = _mm_set1_ps(k);
        size_t f = offset;
        for(; f < tools::round_to_next_multiple(offset,4); ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + k*tanh(z1[c*NF_+f]);

        for(; f < tools::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));

            __m128 v = z2 + phi_signs*vtanh(z2);
            _mm_store_pd(&phi[c*NF_+f],_mm_cvtps_pd(v));
            _mm_store_pd(&phi[c*NF_+f+2],_mm_cvtps_pd(_mm_movehl_ps(v,v)));
        }
        for(; f < offset+sample_size ; ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + k*tanh(z1[c*NF_+f]);

    }
}


template<>
void extended_infomax_ica<double>::compute_dphi(size_t offset, size_t sample_size, double * z1, int const * signs, double* dphi) const {
    #pragma omp parallel for
    for(size_t c = 0 ; c < NC_ ; ++c){
        int s = signs[c];
        size_t f = offset;
        for(; f < tools::round_to_next_multiple(offset,4); ++f){
          double y = tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
        for(; f < tools::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));
            __m128 y = vtanh(z2);
            __m128 val;
            if(s>0)
                val =  _mm_set1_ps(2) - y*y;
            else
                val = y*y;
            _mm_store_pd(&dphi[c*NF_+f],_mm_cvtps_pd(val));
            _mm_store_pd(&dphi[c*NF_+f+2],_mm_cvtps_pd(_mm_movehl_ps(val,val)));
        }
        for(; f < offset+sample_size ; ++f){
          double y = tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
    }
}

template<>
void extended_infomax_ica<double>::compute_means_logp(size_t offset, size_t sample_size, double * z1, int const * signs, double* means_logp) const {
    #pragma omp parallel for
    for(size_t c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        float k = signs[c];
        double sum = 0;
        size_t f = offset;
        for(; f < tools::round_to_next_multiple(offset,4); ++f){
          double z = z1[c*NF_+f];
          sum+=(k<0)? - 0.693147 - 0.5*(z-1)*(z-1) + log(1+exp(-2*z)):-log(cosh(z))-0.5*z*z;
        }
        for(; f < tools::round_to_previous_multiple(offset+sample_size,4) ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 _0_5 = _mm_set1_ps(0.5);
            const __m128 _2 = _mm_set1_ps(2);
            __m128 val;
            if(k<0)
                val = _mm_set1_ps(-0.693147) - _0_5*(z2 - _1)*(z2 - _1) + vlog(_1 + vexp(-_2*z2));
            else
                val = - vlog(vcosh(z2)) - _0_5*z2*z2;
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(val));
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(val,val)));
        }
        for(; f < offset+sample_size; ++f){
          double z = z1[c*NF_+f];
          sum+=(k<0)? - 0.693147 - 0.5*(z-1)*(z-1) + log(1+exp(-2*z)):-log(cosh(z))-0.5*z*z;
        }
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(double)sample_size*sum;
    }
}

}
