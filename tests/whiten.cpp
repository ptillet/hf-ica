#include <stdio.h>
#include <stdlib.h>
#include "viennacl/matrix.hpp"
#include "matrix_io.hpp"
#include "clica.h"

typedef double NumericT;

int main(){
    viennacl::matrix<NumericT, viennacl::row_major> mixed_signals;
    viennacl::matrix<NumericT, viennacl::row_major> true_whitened_data;
    viennacl::matrix<NumericT, viennacl::row_major> clica_whitened_data;
    read_mtx(mixed_signals,"../tests/data/mixed_signals.mtx");
    read_mtx(true_whitened_data,"../tests/data/original_signals.mtx");
    clica::whiten(mixed_signals, clica_whitened_data);
}
