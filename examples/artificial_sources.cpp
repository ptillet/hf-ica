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


typedef double NumericT;

static const unsigned int C=3;
static const unsigned int N=1000;
static const unsigned int T=10;

int main(){
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic> c_src(C,N);
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic> mixing(C,C);
    for(unsigned int i=0 ; i< N ; ++i){
        double t = (double)i/(N-1)*T;
        c_src(0,i) = std::sin(3*t);
        c_src(1,i) = std::cos(10*t);
        c_src(2,i) = std::cos(5*t) + std::sin(2*t);
    }
    mixing << 0.1, 0.3, 0.2,
              0.5, 0.2, 0.4,
              0.1, 0.8, 0.3;
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic> data = mixing*c_src;
    Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic> independent_components(C,N);
    clica::inplace_linear_ica(data,independent_components);
    plot(independent_components);
}
