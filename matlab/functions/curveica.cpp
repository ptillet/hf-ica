#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <cstring>
#include "curveica.h"


inline bool are_string_equal(const char * a, const char * b){
    return std::strcmp(a,b)==0;
}

struct curveica_options_type{
    bool use_float;
    curveica::options opts;
};


void fill_options(mxArray* options_mx, curveica_options_type & options){
    /*-RS-*/
    if(mxArray * RS = mxGetField(options_mx,0, "RS")){
        options.opts.RS = mxGetScalar(RS);
    }

    /*-S0-*/
    if(mxArray * S0 = mxGetField(options_mx,0, "S0")){
        options.opts.S0 = mxGetScalar(S0);
    }

    /*-Max Iter-*/
    if(mxArray * theta = mxGetField(options_mx,0, "theta")){
        options.opts.theta = mxGetScalar(theta);
    }

    /*-Max Iter-*/
    if(mxArray * max_iter = mxGetField(options_mx,0, "maxIter"))
        options.opts.max_iter = mxGetScalar(max_iter);

    /*-Verbosity Level-*/
    if(mxArray * verbosity_level = mxGetField(options_mx,0, "verbosityLevel")){
        options.opts.verbosity_level = mxGetScalar(verbosity_level);
    }

    /*-Direction-*/
   if(mxArray * direction = mxGetField(options_mx,0,"optimizationMethod")){
       char* direction_name = mxArrayToString(direction);
       if(are_string_equal(direction_name,"HESSIAN_FREE"))
           options.opts.optimization_method = curveica::HESSIAN_FREE;
       if(are_string_equal(direction_name,"SD"))
           options.opts.optimization_method = curveica::SD;
       if(are_string_equal(direction_name,"NCG"))
           options.opts.optimization_method = curveica::NCG;
       if(are_string_equal(direction_name,"LBFGS"))
           options.opts.optimization_method = curveica::LBFGS;
       if(are_string_equal(direction_name,"BFGS"))
           options.opts.optimization_method = curveica::BFGS;
   }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    curveica_options_type options;
    //Set default
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


    //Get dimensions
    const mwSize * dims = mxGetDimensions(prhs[0]);
    std::size_t NC = static_cast<std::size_t>(dims[0]);
    std::size_t NF = static_cast<std::size_t>(dims[1]);

    mxArray* weights_tmp = NULL;
    double * weights = NULL;
    if(nlhs>=2){
        weights_tmp = plhs[1] = mxCreateDoubleMatrix(NC,NC,mxREAL);
        weights = mxGetPr(weights_tmp);
    }

    mxArray* sphere_tmp = NULL;
    double * sphere = NULL;
    if(nlhs>=3){
        sphere_tmp = plhs[2] = mxCreateDoubleMatrix(NC,NC,mxREAL);
        sphere = mxGetPr(sphere_tmp);
    }
    if(nlhs>=4)
        mexErrMsgIdAndTxt( "curveica:TooManyOutputArguments",
                           "Too many output arguments : independent_components = linear_curveica(data [, options])");


    if(mxIsDouble(prhs[0])){
        //Get data
        double* data = mxGetPr(prhs[0]);
        mxArray* result_tmp = plhs[0] = mxCreateDoubleMatrix(NC,NF,mxREAL);
        double* result = mxGetPr(result_tmp);

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
    else{
        //Get data
        float * data = (float*)mxGetPr(prhs[0]);
        mxArray* result_tmp = plhs[0] = mxCreateNumericMatrix(NC,NF,mxSINGLE_CLASS, mxREAL);
        float * result = (float*)mxGetPr(result_tmp);

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
}
