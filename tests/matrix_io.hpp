#ifndef PARICA_TESTS_READMATRIX_HPP_
#define PARICA_TESTS_READMATRIX_HPP_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * CLICA - Hybrid ICA using ViennaCL + Eigen
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <string>

#include "boost/numeric/ublas/matrix.hpp"

template<class MatrixT>
bool read_mtx (MatrixT & output  , const std::string & path  )
{
    std::ifstream is (path.c_str() );
    if ( !is )
    {
        std::cerr << "Opening file failure" << std::endl;
        return false;
    }
    else
    {
        std::string str;
        unsigned int size1;
        unsigned int size2;
        double value;
        std::stringstream line ( std::stringstream::in | std::stringstream::out );
        do{ std::getline(is, str); }
        while ( str[0]=='%' );
        line << str;
        line >> size1 >> size2;
        output.resize ( size1,size2,false );
        boost::numeric::ublas::matrix<float> tmp ( size1,size2 );
        for ( unsigned int j=0 ; j<size2 ; ++j )
        {
            for ( unsigned int i=0 ; i<size1 ; ++i )
            {
                is >> value;
                tmp ( i,j ) = value;
            }
        }
        viennacl::copy ( tmp,output );
        return true;
    }
}

#endif
