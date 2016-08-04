/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * DSHF-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef DSHF_ICA_TOOLS_MEX_HPP_
#define DSHF_ICA_TOOLS_MEX_HPP_

#include <cstddef>
#include <tr1/random>

#ifdef USE_MEX
    #include "mex.h"
    #ifdef __cplusplus
        extern "C" bool utIsInterruptPending();
    #else
        extern bool utIsInterruptPending();
    #endif
#endif

namespace dshf_ica
{

class exception : public std::exception
{
public:
  exception() : message_() {}
  exception(std::string message) : message_("DSHF-ICA: " + message) {}
  virtual const char* what() const throw() { return message_.c_str(); }
  virtual ~exception() throw() {}
private:
  std::string message_;
};

inline void throw_if_mex_and_ctrl_c(){
#ifdef USE_MEX
    if (utIsInterruptPending()) {
        throw dshf_ica::exception("Ctrl-C Pressed : Aborting...");
    }
#endif
}

}

#endif
