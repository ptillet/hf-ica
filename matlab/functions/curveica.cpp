#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <cstring>
#include "curveica.h"
#include <Eigen/Dense>

typedef double ScalarType;

inline bool are_string_equal(const char * a, const char * b){
    return std::strcmp(a,b)==0;
}

struct curveica_options_type{
    bool use_float;
    curveica::options opts;
};


void fill_options(mxArray* options_mx, curveica_options_type & options){
    /*-Max Iter-*/
    if(mxArray * use_float = mxGetField(options_mx,0, "useFloat")){
        options.use_float = static_cast<bool>(mxGetScalar(use_float));
    }

    /*-Max Iter-*/
    if(mxArray * max_iter = mxGetField(options_mx,0, "maxIter"))
        options.opts.max_iter = mxGetScalar(max_iter);

    /*-Verbosity Level-*/
    if(mxArray * verbosity_level = mxGetField(options_mx,0, "verbosityLevel")){
        options.opts.verbosity_level = mxGetScalar(verbosity_level);
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    curveica_options_type options;
    //Set default
    options.use_float = true;
    options.opts= curveica::make_default_options();


    if(nrhs>2)
        mexErrMsgIdAndTxt( "curveica:invalidNumInputs",
                           "Invalid input arguments : independent_components = linear_curveica(data [, options])");
    if(nrhs>1){
        if(mxIsStruct(prhs[1])){
            mxArray * options_mx = mxDuplicateArray(prhs[1]);
            fill_options(options_mx, options);
        }
        else
            mexErrMsgIdAndTxt("curveica:invalidOptionsType",
                              "Invalid input arguments : The options must be a valid struct");
    }
    if(nlhs>1)
        mexErrMsgIdAndTxt( "curveica:TooManyOutputArguments",
                           "Too many output arguments : independent_components = linear_curveica(data [, options])");
    //Get data
    mxArray * data_tmp = mxDuplicateArray(prhs[0]);
    ScalarType * data = mxGetPr(data_tmp);

    //Get dimensions
    const mwSize * dims = mxGetDimensions(prhs[0]);
    std::size_t NC = static_cast<std::size_t>(dims[0]);
    std::size_t NF = static_cast<std::size_t>(dims[1]);

    mxArray* result_tmp = plhs[0] = mxCreateDoubleMatrix(NC,NF,mxREAL);
    ScalarType * result = mxGetPr(result_tmp);


    Eigen::Map<Eigen::MatrixXd> map_data(data,NC,NF);
    Eigen::Map<Eigen::MatrixXd> map_result(result,NC,NF);
    if(options.use_float){
        typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixType;
        MatrixType data_eigen = map_data.cast<float>();
        MatrixType result_eigen(NC,NF);
        curveica::inplace_linear_ica(data_eigen.data(), result_eigen.data(),NC,NF,options.opts);
        map_result = result_eigen.cast<double>();
    }
    else{
        typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixType;
        MatrixType data_eigen = map_data;
        MatrixType result_eigen(NC,NF);
        curveica::inplace_linear_ica(data_eigen.data(), result_eigen.data(),NC,NF,options.opts);
        map_result = result_eigen;
    }
}
