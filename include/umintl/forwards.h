#ifndef UMINTL_FORWARDS_H
#define UMINTL_FORWARDS_H

#include <cstddef>
#include "umintl/tools/shared_ptr.hpp"

namespace umintl{


template<class BackendType>
class optimization_context;
template<class BackendType>
struct model_base;

enum computation_type{ CENTERED_DIFFERENCE, FORWARD_DIFFERENCE, PROVIDED };

enum model_type_tag {  DETERMINISTIC, STOCHASTIC };

struct operation_tag {
    operation_tag(model_type_tag const & _model, size_t _sample_size, size_t _offset) : model(_model), sample_size(_sample_size), offset(_offset){ }
    model_type_tag model;
    size_t sample_size;
    size_t offset;
};

struct value_gradient : public operation_tag {
    value_gradient(model_type_tag const & _model, size_t _sample_size, size_t _offset) : operation_tag(_model,_sample_size,_offset){ }
};
struct hessian_vector_product : public operation_tag {
    hessian_vector_product(model_type_tag const & _model, size_t _sample_size, size_t _offset) : operation_tag(_model,_sample_size,_offset){ }
};
struct gradient_variance : public operation_tag {
    gradient_variance(model_type_tag const & _model, size_t _sample_size, size_t _offset) : operation_tag(_model,_sample_size,_offset){ }
};
struct hv_product_variance : public operation_tag {
    hv_product_variance(model_type_tag const & _model, size_t _sample_size, size_t _offset) : operation_tag(_model,_sample_size,_offset){ }
};

}
#endif
