/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef NEO_ICA_TOOLS_SHUFFLE_HPP_
#define NEO_ICA_TOOLS_SHUFFLE_HPP_

#include <cstddef>
#include <random>

namespace neo_ica
{

template<class ScalarType>
void shuffle(ScalarType* data, size_t NC, size_t NF){
    size_t* perms = new size_t[NF];
    ScalarType * shuffled_va = new ScalarType[NF];

    std::minstd_rand gen(0);
    for(size_t i = 0 ; i < NF ; ++i)
        perms[i] = i;
    for(size_t i = 0 ; i < NF ; ++i){
        size_t j = std::uniform_int_distribution<size_t>(i, NF-1)(gen);
        std::swap(perms[i], perms[j]);
    }
    for(size_t c = 0 ; c < NC ; ++c){
        for(size_t f = 0 ; f < NF ; ++f)
            shuffled_va[f] = data[c*NF+perms[f]];
        for(size_t f = 0 ; f < NF ; ++f)
            data[c*NF+f] = shuffled_va[f];
    }

    delete[] shuffled_va;
    delete[] perms;
}

}

#endif
