#ifndef CLICA_H_
#define CLICA_H_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

namespace clica{

template<class T, class U>
void whiten(T & data, U & out);

template<class T, class U>
void inplace_linear_ica(T & in, U & out);

}

#endif
