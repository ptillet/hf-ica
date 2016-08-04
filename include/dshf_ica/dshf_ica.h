#ifndef dshf_ica_H_
#define dshf_ica_H_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * dshf_ica - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstddef>

namespace dshf_ica{

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
void inplace_linear_ica(ScalarType const * data, ScalarType* W, ScalarType* S, size_t NC, size_t NF, options const & opt);

}

#endif
