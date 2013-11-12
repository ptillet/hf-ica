#include "extended_infomax.h"

#include "tests/benchmark-utils.hpp"

#include <cmath>

#include "omp.h"

#include "src/math/math.h"
#include "src/utils.hpp"

#include <pmmintrin.h>
#include <cstddef>

namespace curveica{

//template<>
//void extended_infomax_ica<float>::operator()(float * z1, float * b, int const * signs, float* phi, float* means_logp) const {
//    for(unsigned int c = 0 ; c < NC_ ; ++c){
//        float current = 0;
//        float k = signs[c];
//        float bias = b[c];
//        for(unsigned int f = 0; f < NF_ ; f++){
//            float z2 = z1[c*NF_+f] + bias;
//            float y = std::tanh(z2);
//            if(k<0){
//                current+= std::log((std::exp(-0.5*std::pow(z2-1,2)) + std::exp(-0.5*std::pow(z2+1,2))));
//                phi[c*NF_+f] = z2 - y;
//            }
//            else{
//                current+= -std::log(std::cosh(z2)) - 0.5*z2*z2;
//                phi[c*NF_+f] = z2 + y;
//            }
//        }
//        means_logp[c] = current/(float)NF_;
//    }
//}

template<>
void extended_infomax_ica<float>::compute_phi(std::size_t offset, std::size_t sample_size, float * z1, int const * signs, float* phi) const {
#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        int s = signs[c];
        __m128 phi_signs = _mm_set1_ps(s);
        unsigned int f = offset;
        for(; f < detail::round_to_next_multiple(offset,4); ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + s*std::tanh(z1[c*NF_+f]);
        for(; f < detail::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);
            //compute phi
            __m128 y = curveica::math::vtanh(z2);
            y = _mm_mul_ps(phi_signs,y);
            __m128 v = _mm_add_ps(z2,y);
            _mm_store_ps(&phi[c*NF_+f],v);
        }
        for(; f < offset+sample_size ; ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + s*std::tanh(z1[c*NF_+f]);
    }

}

template<>
void extended_infomax_ica<float>::compute_dphi(std::size_t offset, std::size_t sample_size, float * z1, int const * signs, float* dphi) const {
#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        int s = signs[c];
        unsigned int f = offset;
        for(; f < detail::round_to_next_multiple(offset,4); ++f){
          float y = std::tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
        for(unsigned int f = offset; f < detail::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);
            __m128 y = curveica::math::vtanh(z2);
            __m128 val;
            if(s>0)
                val =  _mm_set1_ps(2) - y*y;
            else
                val = y*y;
            _mm_store_ps(&dphi[c*NF_+f],val);
        }
        for(; f < offset+sample_size ; ++f){
          float y = std::tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
    }
}

template<>
void extended_infomax_ica<float>::compute_means_logp(std::size_t offset, std::size_t sample_size, float * z1, int const * signs, float* means_logp) const {

#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        int s = signs[c];
        double sum = 0;
        unsigned int f = offset;
        for(; f < detail::round_to_next_multiple(offset,4); ++f){
          float z = z1[c*NF_+f];
          sum+=(s<0)? - 0.693147 - 0.5*(z-1)*(z-1) + std::log(1+std::exp(-2*z)):-std::log(std::cosh(z))-0.5*z*z;
        }
        for(; f < detail::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128 z2 = _mm_load_ps(&z1[c*NF_+f]);

            //Computes mean_logp
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 _0_5 = _mm_set1_ps(0.5);
            const __m128 _2 = _mm_set1_ps(2);
            __m128 val;
            if(s<0)
                val = _mm_set1_ps(-0.693147) - _0_5*(z2 - _1)*(z2 - _1) + curveica::math::vlog(_1 + math::vexp(-_2*z2));
            else
                val = - math::vlog(math::vcosh(z2)) - _0_5*z2*z2;
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(val));
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(val,val)));
        }
        for(; f < offset+sample_size; ++f){
          float z = z1[c*NF_+f];
          sum+=(s<0)? - 0.693147 - 0.5*(z-1)*(z-1) + std::log(1+std::exp(-2*z)):-std::log(std::cosh(z))-0.5*z*z;
        }
        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(double)sample_size*sum;
    }

}


//template<>
//void extended_infomax_ica<double>::operator()(double * z1, double * b, int const * signs, double* phi, double* means_logp) const {
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
void extended_infomax_ica<double>::compute_phi(std::size_t offset, std::size_t sample_size, double * z1, int const * signs, double* phi) const {
//#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        float k = (signs[c]>0)?1:-1;
        __m128 phi_signs = _mm_set1_ps(k);

        unsigned int f = offset;
        for(; f < detail::round_to_next_multiple(offset,4); ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + k*std::tanh(z1[c*NF_+f]);
        for(; f < detail::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));

            __m128 y = math::vtanh(z2);
            y = _mm_mul_ps(phi_signs,y);
            __m128 v = _mm_add_ps(z2,y);
            _mm_store_pd(&phi[c*NF_+f],_mm_cvtps_pd(v));
            _mm_store_pd(&phi[c*NF_+f+2],_mm_cvtps_pd(_mm_movehl_ps(v,v)));
        }
        for(; f < offset+sample_size ; ++f)
          phi[c*NF_+f] = z1[c*NF_+f] + k*std::tanh(z1[c*NF_+f]);

    }
}

template<>
void extended_infomax_ica<double>::compute_dphi(std::size_t offset, std::size_t sample_size, double * z1, int const * signs, double* dphi) const {
#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        int s = signs[c];
        unsigned int f = offset;
        for(; f < detail::round_to_next_multiple(offset,4); ++f){
          double y = std::tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
        for(; f < detail::round_to_previous_multiple(offset+sample_size,4)  ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));
            __m128 y = curveica::math::vtanh(z2);
            __m128 val;
            if(s>0)
                val =  _mm_set1_ps(2) - y*y;
            else
                val = y*y;
            _mm_store_pd(&dphi[c*NF_+f],_mm_cvtps_pd(val));
            _mm_store_pd(&dphi[c*NF_+f+2],_mm_cvtps_pd(_mm_movehl_ps(val,val)));
        }
        for(; f < offset+sample_size ; ++f){
          double y = std::tanh(z1[c*NF_+f]);
          dphi[c*NF_+f] = (s>0)?2-y*y:y*y;
        }
    }
}

template<>
void extended_infomax_ica<double>::compute_means_logp(std::size_t offset, std::size_t sample_size, double * z1, int const * signs, double* means_logp) const {
#pragma omp parallel for
    for(unsigned int c = 0 ; c < NC_ ; ++c){
        __m128d vsum = _mm_set1_pd(0.0d);
        float k = signs[c];
        double sum = 0;
        unsigned int f = offset;
        for(; f < detail::round_to_next_multiple(offset,4); ++f){
          double z = z1[c*NF_+f];
          sum+=(k<0)? - 0.693147 - 0.5*(z-1)*(z-1) + std::log(1+std::exp(-2*z)):-std::log(std::cosh(z))-0.5*z*z;
        }
        for(; f < detail::round_to_previous_multiple(offset+sample_size,4) ; f+=4){
            __m128d z2lo = _mm_load_pd(&z1[c*NF_+f]);
            __m128d z2hi = _mm_load_pd(&z1[c*NF_+f+2]);
            __m128 z2 = _mm_movelh_ps(_mm_cvtpd_ps(z2lo), _mm_cvtpd_ps(z2hi));
            const __m128 _1 = _mm_set1_ps(1);
            const __m128 _0_5 = _mm_set1_ps(0.5);
            const __m128 _2 = _mm_set1_ps(2);
            __m128 val;
            if(k<0)
                val = _mm_set1_ps(-0.693147) - _0_5*(z2 - _1)*(z2 - _1) + curveica::math::vlog(_1 + math::vexp(-_2*z2));
            else
                val = - math::vlog(math::vcosh(z2)) - _0_5*z2*z2;
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(val));
            vsum=_mm_add_pd(vsum,_mm_cvtps_pd(_mm_movehl_ps(val,val)));
        }
        for(; f < offset+sample_size; ++f){
          double z = z1[c*NF_+f];
          sum+=(k<0)? - 0.693147 - 0.5*(z-1)*(z-1) + std::log(1+std::exp(-2*z)):-std::log(std::cosh(z))-0.5*z*z;
        }

        vsum = _mm_hadd_pd(vsum, vsum);
        _mm_store_sd(&sum, vsum);
        means_logp[c] = 1/(double)sample_size*sum;
    }
}

}
