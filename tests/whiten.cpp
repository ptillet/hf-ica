#include "viennacl/matrix.hpp"
#include "matrix_io.hpp"

typedef double NumericT;
static const unsigned int size = 1024;

int main(){
    viennacl::matrix<NumericT> mixed_signals;
    viennacl::matrix<NumericT> whitened_data;
    read_mtx(mixed_signals,"../tests/data/mixed_signals.mtx");
    read_mtx(whitened_data,"../tests/data/whitened_data.mtx");
    std::cout << whitened_data(1,44) << std::endl;
}
