#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <cstring>
#include "curveica.h"

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

    //Get data
    mxArray * data_tmp = mxDuplicateArray(prhs[0]);
    ScalarType * data = mxGetPr(data_tmp);

    //Get dimensions
    const mwSize * dims = mxGetDimensions(prhs[0]);
    std::size_t NC = static_cast<std::size_t>(dims[0]);
    std::size_t NF = static_cast<std::size_t>(dims[1]);

    mxArray* result_tmp = plhs[0] = mxCreateDoubleMatrix(NC,NF,mxREAL);
    ScalarType * result = mxGetPr(result_tmp);

    mxArray* weights_tmp = NULL;
    ScalarType * weights = NULL;
    if(nlhs>=2){
        weights_tmp = plhs[1] = mxCreateDoubleMatrix(NC,NF,mxREAL);
        weights = mxGetPr(weights_tmp);
    }

    mxArray* sphere_tmp = NULL;
    ScalarType * sphere = NULL;
    if(nlhs>=3){
        sphere_tmp = plhs[2] = mxCreateDoubleMatrix(NC,NF,mxREAL);
        sphere = mxGetPr(sphere_tmp);
    }
    if(nlhs>=4)
        mexErrMsgIdAndTxt( "curveica:TooManyOutputArguments",
                           "Too many output arguments : independent_components = linear_curveica(data [, options])");



    if(options.use_float){
        float* data_float = new float[NC*NF];
        float* result_float = new float[NC*NF];
        for(std::size_t c = 0 ; c < NC; ++c)
            for(std::size_t f = 0 ; f < NF ; ++f)
                data_float[c*NF+f] = data[f*NC+c];

        curveica::inplace_linear_ica(data_float, result_float,NC,NF,options.opts,weights,sphere);

        for(std::size_t c = 0 ; c < NC; ++c)
            for(std::size_t f = 0 ; f < NF ; ++f)
                result[f*NC+c] = result_float[c*NF+f];

        delete[] data_float;
        delete[] result_float;
    }
    else{
        double* data_double = new double[NC*NF];
        double* result_double = new double[NC*NF];
        for(std::size_t c = 0 ; c < NC; ++c)
            for(std::size_t f = 0 ; f < NF ; ++f)
                data_double[c*NF+f] = data[f*NC+c];

        curveica::inplace_linear_ica(data_double, result_double,NC,NF,options.opts,weights,sphere);

        for(std::size_t c = 0 ; c < NC; ++c)
            for(std::size_t f = 0 ; f < NF ; ++f)
                result[f*NC+c] = result_double[c*NF+f];

        delete[] data_double;
        delete[] result_double;
    }
}
