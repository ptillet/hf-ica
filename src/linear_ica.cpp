#include "clica.h"

#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

#include "Eigen/Dense"

namespace clica{

template<class MAT>
struct ica_functor{
private:
    typedef typename MAT::value_type::value_type NumericT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> CPU_MAT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic, 1> CPU_VEC;
public:
    ica_functor(MAT const & data) : data_(data){ }

    NumericT operator()(CPU_VEC const & x, CPU_VEC * grad) const {
        size_t nchans = data_.size1();
        size_t nframes = data_.size2();
        CPU_MAT weights(nchans, nchans);
        CPU_VEC bias(nchans);
        std::memcpy(weights.data(), x.data(),sizeof(NumericT)*nchans*nchans);
        std::memcpy(bias.data(), x.data()+nchans*nchans, sizeof(NumericT)*nchans);
        std::cout << weights << std::endl;
        std::cout << bias << std::endl;

//        viennacl::copy(x.begin()+nchans*nchans+1, x.end(), bias);

    }

private:
    MAT const & data_;
};

template<class MAT>
void inplace_linear_ica(MAT & data, MAT & out){
    typedef typename MAT::value_type::value_type NumericT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic,Eigen::Dynamic> CPU_MAT;
    typedef Eigen::Matrix<NumericT,Eigen::Dynamic, 1> CPU_VEC;

    size_t nchans = data.size1();
    size_t nframes = data.size2();

    MAT white_data(nchans, nframes);
    whiten(data,white_data);
    CPU_VEC X = CPU_VEC::Zero(nchans*nchans + nchans);
    for(unsigned int i = 0 ; i < nchans; ++i) X[i*(nchans+1)] = 1;
    for(unsigned int i = nchans*nchans ; i < nchans*(nchans+1) ; ++i) X[i] = 2;
    ica_functor<MAT> fun(white_data);
    fun(X,NULL);

}

template void inplace_linear_ica<viennacl::matrix<double, viennacl::row_major> >(viennacl::matrix<double, viennacl::row_major> &, viennacl::matrix<double, viennacl::row_major> &);

}

