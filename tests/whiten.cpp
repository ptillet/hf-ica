#include <stdio.h>
#include <stdlib.h>
#include "viennacl/matrix.hpp"
#include "matrix_io.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "clica.h"

#define DATA_PATH "/home/philippe/Development/CLICA/tests/data/"

typedef double NumericT;

template<class MAT>
double diff(MAT const & mat1, MAT const & mat2){
    size_t size = mat1.internal_size();
    std::vector<NumericT> c_mat1(size);
    std::vector<NumericT> c_mat2(size);
    viennacl::fast_copy(mat1, c_mat1.data());
    viennacl::fast_copy(mat2, c_mat2.data());
    viennacl::backend::finish();
    double res = 0;
    for(unsigned int i = 0; i<size ; ++i){
        res = std::max(res, static_cast<double>(std::fabs(c_mat1[i]-c_mat2[i])/std::max(c_mat1[i], c_mat2[i])));
    }
    return res;
}

int main(){
    double tol = 1e-4;
    viennacl::matrix<NumericT, viennacl::row_major> mixed_signals;
    viennacl::matrix<NumericT, viennacl::row_major> true_whitened_data;
    read_mtx(mixed_signals,DATA_PATH "mixed_signals.mtx");
    read_mtx(true_whitened_data,DATA_PATH "white_data.mtx");
    viennacl::matrix<NumericT, viennacl::row_major> clica_whitened_data(mixed_signals.size1(), mixed_signals.size2());
    clica::whiten(mixed_signals, clica_whitened_data);
    return diff(clica_whitened_data, true_whitened_data) > tol;
}
