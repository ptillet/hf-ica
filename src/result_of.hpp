/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef PARICA_RESULT_OF_HPP_
#define PARICA_RESULT_OF_HPP_


namespace parica{

namespace result_of{

template<class ScalarType>
struct weights{
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> type;
};

template<class ScalarType>
struct data_storage{
    //We consider we have one channel per row. Storing the data in row-major ensure better cache behavior and higher bandwidth
    //It allows for example one core to process one channel without generating too many conflicts
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> type;
};


}

}

#endif
