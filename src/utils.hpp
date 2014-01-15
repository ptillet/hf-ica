/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * DSHF-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef dshf_ica_UTILS_HPP_
#define dshf_ica_UTILS_HPP_

#include <cstddef>
#include <tr1/random>

#ifdef USE_MEX
    #include "mex.h"
    #ifdef __cplusplus
        extern "C" bool utIsInterruptPending();
    #else
        extern bool utIsInterruptPending();
    #endif
#endif
namespace dshf_ica{

    class exception : public std::exception
    {
    public:
      exception() : message_() {}
      exception(std::string message) : message_("DSHF-ICA: " + message) {}
      virtual const char* what() const throw() { return message_.c_str(); }
      virtual ~exception() throw() {}
    private:
      std::string message_;
    };


    inline void throw_if_mex_and_ctrl_c(){
#ifdef USE_MEX
        if (utIsInterruptPending()) {
            throw dshf_ica::exception("Ctrl-C Pressed : Aborting...");
        }
#endif

    }

    namespace detail{

        template<class ScalarType>
        std::size_t shuffle(ScalarType* data, std::size_t NC, std::size_t NF){
            std::size_t* perms = new std::size_t[NF];
            ScalarType * shuffled_va = new ScalarType[NF];

            std::tr1::minstd_rand gen;
//            //gen.seed(time(NULL));

            for(std::size_t i = 0 ; i < NF ; ++i)
                perms[i] = i;
            for(std::size_t i = 0 ; i < NF ; ++i){
                std::size_t j = std::tr1::uniform_int<std::size_t>(i, NF-1)(gen);
                std::swap(perms[i], perms[j]);
            }
            for(std::size_t c = 0 ; c < NC ; ++c){
                for(std::size_t f = 0 ; f < NF ; ++f)
                    shuffled_va[f] = data[c*NF+perms[f]];
                for(std::size_t f = 0 ; f < NF ; ++f)
                    data[c*NF+f] = shuffled_va[f];
            }

            delete[] shuffled_va;
            delete[] perms;
        }

        template<class T>
        T round_to_previous_multiple(T const & val, unsigned int multiple){
          return val/multiple*multiple;
        }

        template<class T>
        T round_to_next_multiple(T const & val, unsigned int multiple){
          return (val+multiple-1)/multiple*multiple;
        }
    }

}

#endif
