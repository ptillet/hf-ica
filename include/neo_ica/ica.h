#ifndef neo_ica_H_
#define neo_ica_H_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * neo_ica - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstddef>
#include <cstdint>

namespace neo_ica{

struct options{
    size_t max_iter;
    unsigned int verbosity_level;
    double theta;
    double RS;
    size_t S0;
    int omp_num_threads;
};

options make_default_options();

template<class ScalarType>
void ica(ScalarType const * data, ScalarType* W, ScalarType* S, int64_t NC, int64_t NF, options const & opt);

}

#endif
