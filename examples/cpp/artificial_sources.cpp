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

#define BENCHMARK_COUNT 5

typedef double NumericT;
typedef Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatType;
static const unsigned int C=4;
static const unsigned int N=1000;
static const unsigned int T=20;

int main(){
    MatType c_src(C,N);
    for(unsigned int i=0 ; i< N ; ++i){
        double t = (double)i/(N-1)*T - T/2;
        c_src(0,i) = std::sin(3*t) + std::cos(6*t);
        c_src(1,i) = std::cos(10*t);
        c_src(2,i) = std::sin(5*t);
        c_src(3,i) = std::sin(t*t);
    }
    MatType mixing(C,C);
    for(std::size_t i = 0 ; i < C ; ++i)
		for(std::size_t j = 0 ; j < C ; ++j)
			mixing(i,j) = static_cast<double>(rand())/RAND_MAX;
    MatType data = mixing*c_src;
    MatType independent_components(C,N);
    fmincl::optimization_options options = parica::make_default_options();
    options.verbosity_level = 0;
    Timer t;
    t.start();
    for(unsigned int i = 0 ; i < BENCHMARK_COUNT ; ++i)
        parica::inplace_linear_ica(data,independent_components,options);
    std::cout << "Execution Time : " << t.get()/BENCHMARK_COUNT << "s" << std::endl;
}
