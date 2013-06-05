/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cmath>
#include "viennacl/matrix.hpp"
#include "viennacl/rand/uniform.hpp"
#include "viennacl/linalg/prod.hpp"

#include "tests/matrix_io.hpp"

#include "clica.h"
#include "plot.hpp"

#define DATA_PATH "/home/philippe/Development/CLICA/tests/data/"

typedef double NumericT;

static const unsigned int C=4;
static const unsigned int N=1000;
static const unsigned int T=10;

int main(){
    std::vector<std::vector<NumericT> > c_src(C);
    for(unsigned int i=0 ; i< N ; ++i){
        double t = (double)i/N*T;
        c_src[0].push_back(std::sin(3*t));
        c_src[1].push_back(std::cos(10*t));
        c_src[2].push_back(std::sin(5*t)*std::cos(3*t));
        c_src[3].push_back(rand()/(double)RAND_MAX);
    }
    viennacl::matrix<NumericT> mixing(C,C);
    viennacl::matrix<NumericT> src(C,N);
    viennacl::copy(c_src,src);
    for(unsigned int i = 0 ; i < C ; ++i){
        for(unsigned int j = 0 ; j < C ; ++j){
            mixing(i,j) = (NumericT)rand()/RAND_MAX;
        }
    }
    viennacl::matrix<NumericT> data = viennacl::linalg::prod(mixing,src);


    read_mtx(data,DATA_PATH "mixed_signals.mtx");
    viennacl::matrix<NumericT> independent_components(data.size1(),data.size2());
    clica::inplace_linear_ica(data,independent_components);
    plot(independent_components);
}
