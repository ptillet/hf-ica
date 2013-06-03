#ifndef CLICA_EXAMPLES_HPP_
#define CLICA_EXAMPLES_HPP_

#include <gnuplot-cpp/gnuplot_i.hpp>
#include <sstream>

template<class NumericT>
void plot(std::vector<std::vector<NumericT> > const & rows){
	size_t n = rows.size();
	Gnuplot g1("lines");
	g1.set_multiplot(n,1);
    for(unsigned int i = 0 ; i < n ; ++i){
		std::ostringstream oss;
		oss << "Row " << i;
		g1.plot_x(rows[i],oss.str());
	}
	std::cout << std::endl << "Press ENTER to continue..." << std::endl;
	std::cin.clear();
	std::cin.ignore(std::cin.rdbuf()->in_avail());
	std::cin.get();
}

template<class NumericT>
void plot(viennacl::matrix<NumericT, viennacl::row_major> const & mat){
	std::vector<std::vector<NumericT> > rows;
	for(unsigned int i=0 ; i < mat.size1() ; ++i)
		rows.push_back(std::vector<NumericT>(mat.size2()));
	viennacl::copy(mat, rows);
	plot(rows);
}

#endif
