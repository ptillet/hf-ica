/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef NEO_ICA_TOOLS_MEX_HPP_
#define NEO_ICA_TOOLS_MEX_HPP_

#include <cstddef>
#include <exception>
#include <string>

#ifdef USE_MEX
    #include "mex.h"
    #ifdef __cplusplus
        extern "C" bool utIsInterruptPending();
    #else
        extern bool utIsInterruptPending();
    #endif
#endif

namespace neo_ica
{

class exception : public std::exception
{
public:
  exception() : message_() {}
  exception(std::string message) : message_("NEO-ICA: " + message) {}
  virtual const char* what() const throw() { return message_.c_str(); }
  virtual ~exception() throw() {}
private:
  std::string message_;
};

inline void throw_if_mex_and_ctrl_c(){
#ifdef USE_MEX
    if (utIsInterruptPending()) {
        throw neo_ica::exception("Ctrl-C Pressed : Aborting...");
    }
#endif
}

}

#endif
