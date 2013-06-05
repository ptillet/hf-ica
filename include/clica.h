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

template<class MAT>
void whiten(MAT & data, MAT & out);

template<class MAT>
void inplace_linear_ica(MAT & in, MAT & out);

}

#endif
