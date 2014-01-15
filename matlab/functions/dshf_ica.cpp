/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * DSHF-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <cstring>
#include <streambuf>
#include <iostream>
#include "dshf_ica.h"


class mstream : public std::streambuf {
public:
protected:
  virtual std::streamsize xsputn(const char *s, std::streamsize n){
        mexPrintf("%.*s",n,s);
        mexEvalString("pause(.0001);"); // to dump string.
        return n;
  }
  virtual int overflow(int c = EOF){
        if (c != EOF) {
          mexPrintf("%.1s",&c);
        }
        return c;
  }
};


inline bool are_string_equal(const char * a, const char * b){
    return std::strcmp(a,b)==0;
}

struct dshf_ica_options_type{
    bool use_float;
    dshf_ica::options opts;
};


template<class ScalarType>
inline void transpose(ScalarType *m, int w, int h)
{
  int start, next, i;
  double tmp;

  for (start = 0; start <= w * h - 1; start++) {
    next = start;
    i = 0;
    do {	i++;
      next = (next % h) * w + next / h;
    } while (next > start);
    if (next < start || i == 1) continue;

    tmp = m[next = start];
    do {
      i = (next % h) * w + next / h;
      m[next] = (i == start) ? tmp : m[i];
      next = i;
    } while (next > start);
  }
}

void fill_options(mxArray* options_mx, dshf_ica_options_type & options){
    /*-RS-*/
    if(mxArray * RS = mxGetField(options_mx,0, "RS")){
        options.opts.RS = mxGetScalar(RS);
    }

    /*-S0-*/
    if(mxArray * S0 = mxGetField(options_mx,0, "S0")){
        options.opts.S0 = mxGetScalar(S0);
    }

    /*-Theta-*/
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
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    dshf_ica_options_type options;
    //Set default
    options.opts= dshf_ica::make_default_options();
    mstream mout;
    if(options.opts.verbosity_level>0){
        std::cout.rdbuf(&mout);
    }

    if(nrhs>2)
        mexErrMsgIdAndTxt( "dshf_ica:invalidNumInputs",
                           "Invalid input arguments : independent_components = linear_dshf_ica(data [, options])");
    if(nrhs>1){
        if(mxIsStruct(prhs[1])){
            mxArray * options_mx = mxDuplicateArray(prhs[1]);
            fill_options(options_mx, options);
        }
        else
            mexErrMsgIdAndTxt("dshf_ica:invalidOptionsType",
                              "Invalid input arguments : The options must be a valid struct");
    }


    //Get dimensions
    const mwSize * dims = mxGetDimensions(prhs[0]);
    std::size_t NC = static_cast<std::size_t>(dims[0]);
    std::size_t NF = static_cast<std::size_t>(dims[1]);

    mxArray* weights_tmp = NULL;
    double * weights = NULL;
    weights_tmp = plhs[0] = mxCreateDoubleMatrix(NC,NC,mxREAL);
    weights = mxGetPr(weights_tmp);

    mxArray* sphere_tmp = NULL;
    double * sphere = NULL;
    sphere_tmp = plhs[1] = mxCreateDoubleMatrix(NC,NC,mxREAL);
    sphere = mxGetPr(sphere_tmp);

    if(nlhs!=2)
        mexErrMsgIdAndTxt( "dshf_ica:WrongOutputArguments",
                           "Usage : [W, Sphere] = dshf_ica(data [, options])");


    if(mxIsDouble(prhs[0])){
        //Get data
        double* data = mxGetPr(prhs[0]);
        transpose(data,NC,NF);

        dshf_ica::inplace_linear_ica(data, weights, sphere, NC, NF, options.opts);

        transpose(weights,NC,NC);
        transpose(sphere,NC,NC);
        transpose(data,NF,NC);
    }
    else{
        //Get data
        float * data = (float*)mxGetPr(prhs[0]);
        float * weights_float = new float[NC*NC];
        float * sphere_float = new float[NC*NC];
        transpose(data,NC,NF);

        dshf_ica::inplace_linear_ica(data, weights_float, sphere_float, NC, NF, options.opts);

        for(std::size_t i = 0 ; i < NC ; ++i){
            for(std::size_t j = 0 ; j < NC ; ++j){
                weights[i*NC+j] = weights_float[j*NC+i];
                sphere[i*NC+j] = sphere_float[j*NC+i];
            }
        }
        transpose(data,NF,NC);
    }
}
