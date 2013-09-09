#ifndef CLICA_H_
#define CLICA_H_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include "fmincl/optimization_otions.hpp"
#include "Eigen/Dense"

namespace parica{

namespace result_of{

template<class ScalarType>
struct internal_matrix_type{
    //We consider we have one channel per row. Storing the data in row-major ensure better cache behavior and higher bandwidth
    //It allows for example one core to process one channel without generating too many cache conflicts
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> type;
};

template<class ScalarType>
struct internal_vector_type{
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1 > type;
};


}

fmincl::optimization_options make_default_options();

template<class ScalarType>
void inplace_linear_ica(ScalarType const * data, ScalarType * out, std::size_t NC, std::size_t NF, fmincl::optimization_options const & options);

}

#endif
