#ifndef neo_ica_H_
#define neo_ica_H_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * neo_ica - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstddef>
#include <cstdint>
#include <string>

namespace neo_ica{


namespace dflt{
    static const size_t iter = 500;
    static const unsigned int verbose = 0;
    static const double theta = 0.5;
    static const double rho = 0.25;
    static const size_t fbatch = 1024;
    static const int nthreads = 0;
    static const double tol = 1e-5;
    static const bool extended = true;
}

struct options{
    options(size_t _iter = dflt::iter,
            unsigned int _verbose = dflt::verbose,
            double _theta = dflt::theta,
            double _rho = dflt::rho,
            double _fbatch = dflt::fbatch,
            double _nthreads = dflt::nthreads,
            bool _extended = dflt::extended,
            double _tol = dflt::tol):
        iter(_iter), verbose(_verbose), theta(_theta), rho(_rho),
        fbatch(_fbatch), nthreads(_nthreads), extended(_extended), tol(_tol){}

    size_t iter;
    unsigned int verbose;
    double theta;
    double rho;
    size_t fbatch;
    int nthreads;
    bool extended;
    double tol;
};

template<class ScalarType>
void ica(ScalarType const * data, ScalarType* W, ScalarType* S, int64_t NC, int64_t NF, options const & opt = options());

}

#endif
