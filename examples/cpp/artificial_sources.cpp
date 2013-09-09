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
#include "parica.h"
#include "tests/benchmark-utils.hpp"
#include <cstdlib>

#define BENCHMARK_COUNT 1

typedef float NumericT;
typedef Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatType;
static const unsigned int NC=4;
static const unsigned int NF=1000;
static const unsigned int T=20;

int main(){
    MatType c_src(NC,NF);
    for(unsigned int f=0 ; f< NF ; ++f){
        double t = (double)f/(NF-1)*T - T/2;
        c_src(0,f) = std::sin(3*t) + std::cos(6*t);
        c_src(1,f) = std::cos(10*t);
        c_src(2,f) = std::sin(5*t);
        c_src(3,f) = std::sin(t*t);
    }
    MatType mixing(NC,NC);
    for(std::size_t i = 0 ; i < NC ; ++i)
        for(std::size_t j = 0 ; j < NC ; ++j)
			mixing(i,j) = static_cast<double>(rand())/RAND_MAX;
    MatType mixed_src = mixing*c_src;
    MatType independent_components(NC,NF);
    fmincl::optimization_options options = parica::make_default_options();
    options.verbosity_level = 2;
    Timer t;
    t.start();
    for(unsigned int i = 0 ; i < BENCHMARK_COUNT ; ++i)
        parica::inplace_linear_ica(mixed_src.data(),independent_components.data(),NC,NF,options);
    std::cout << "Execution Time : " << t.get()/BENCHMARK_COUNT << "s" << std::endl;
}
