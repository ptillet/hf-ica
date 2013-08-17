/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cmath>
#include "Eigen/Dense"
#include "clica.h"
#include "plot.hpp"
#include "tests/benchmark-utils.hpp"


typedef double NumericT;

static const unsigned int C=4;
static const unsigned int N=1000;
static const unsigned int T=20;

int main(){
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c_src(C,N);
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mixing(C,C);
    for(unsigned int i=0 ; i< N ; ++i){
        double t = (double)i/(N-1)*T - T/2;
        c_src(0,i) = std::sin(3*t) + std::cos(6*t);
        c_src(1,i) = std::cos(10*t);
        c_src(2,i) = std::sin(5*t);
        c_src(3,i) = std::sin(t*t);
    }
    mixing << 0.1, 0.3, 0.2, 0.04,
              0.5, 0.2, 0.4, 0.8,
              0.1, 0.8, 0.3, 0.2,
              0.05, 0.2, 0.1, 0.3;
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data = mixing*c_src;
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> independent_components(C,N);
    plot(data);
    Timer t;
    t.start();
    clica::inplace_linear_ica(data,independent_components);
    std::cout << t.get() << std::endl;
    plot(independent_components);
}
