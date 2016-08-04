/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * DSHF-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef DSHF_ICA_TOOLS_ROUND_HPP_
#define DSHF_ICA_TOOLS_ROUND_HPP_

#include <cstddef>

namespace dshf_ica
{
namespace tools
{

template<class T>
T round_to_previous_multiple(T const & val, unsigned int multiple)
{ return val/multiple*multiple; }

template<class T>
T round_to_next_multiple(T const & val, unsigned int multiple)
{ return (val+multiple-1)/multiple*multiple; }

}
}

#endif
