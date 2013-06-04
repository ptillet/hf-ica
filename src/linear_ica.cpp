#include "clica.h"

#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

namespace clica{

template<class MAT>
struct ica_functor{
private:
    typedef typename MAT::value_type::value_type NumericT;

public:
    ica_functor(MAT const & data) : data_(data){ }

    NumericT operator()(viennacl::vector<NumericT> const & x, viennacl::vector<NumericT> * grad) const {
        size_t nchans = data_.size1();
        size_t nframes = data_.size2();
        MAT weights(nchans, nchans);
        viennacl::vector<NumericT> bias(nchans);
        viennacl::backend::memory_copy(x.handle(),weights.handle(),0,0,sizeof(NumericT)*nchans*nchans);
        viennacl::backend::memory_copy(x.handle(),bias.handle(),sizeof(NumericT)*nchans*nchans,0,sizeof(NumericT)*nchans);

//        viennacl::copy(x.begin()+nchans*nchans+1, x.end(), bias);

    }

private:
    MAT const & data_;
};

template<class MAT>
void inplace_linear_ica(MAT & data, MAT & out){
    typedef typename MAT::value_type::value_type NumericT;
    MAT white_data(data.size1(), data.size2());
    size_t nchans = data.size1();
    size_t nframes = data.size2();
    MAT weights = viennacl::identity_matrix<NumericT>(nchans);
    viennacl::vector<NumericT> bias = viennacl::scalar_vector<NumericT>(nchans,2);
    viennacl::vector<NumericT> X(nchans*nchans + nchans);

    viennacl::backend::memory_copy(weights.handle(),X.handle(),0,0,sizeof(NumericT)*nchans*nchans);
    viennacl::backend::memory_copy(bias.handle(),X.handle(),0,nchans*nchans*sizeof(NumericT),sizeof(NumericT)*nchans);
    whiten(data,white_data);
    std::cout << weights << std::endl;
    std::cout << X << std::endl;
    ica_functor<MAT> fun(data);
    fun(X,NULL);

}

template void inplace_linear_ica<viennacl::matrix<double, viennacl::row_major> >(viennacl::matrix<double, viennacl::row_major> &, viennacl::matrix<double, viennacl::row_major> &);

}

