/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef CLICA_EXAMPLES_HPP_
#define CLICA_EXAMPLES_HPP_

#include "Eigen/Dense"
#include <gnuplot-cpp/gnuplot_i.hpp>
#include <sstream>

template<class MAT>
void plot(MAT & rows){
    size_t n = rows.rows();
    size_t k = rows.cols();
    Gnuplot g1("lines");
    g1.set_multiplot(n,1);
    for(unsigned int i = 0 ; i < n ; ++i){
		std::ostringstream oss;
		oss << "Row " << i;
        std::vector<double> data(k);
        for(unsigned int j = 0 ; j < k ; ++j) data[j] = rows(i,j);
        g1.plot_x(data,oss.str());
    }
    std::cout << std::endl << "Press ENTER to continue..." << std::endl;
    std::cin.clear();
    std::cin.ignore(std::cin.rdbuf()->in_avail());
    std::cin.get();
}

#endif
