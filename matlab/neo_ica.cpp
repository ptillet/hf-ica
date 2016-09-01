/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * NEO-ICA - Dynamically Sampled Hessian Free Independent Comopnent Analaysis
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <cstring>
#include <string>
#include <streambuf>
#include <iostream>
#include "neo_ica/ica.h"


static std::string USAGE_STR = "Usage : [W, Sphere] = neo_ica(data [, options])";

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

struct neo_ica_options_type{
    bool use_float;
    neo_ica::options opts;
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

void fill_options(mxArray* options_mx, neo_ica_options_type & options){
    if(mxArray * rho = mxGetField(options_mx,0, "rho"))
        options.opts.rho = mxGetScalar(rho);
    if(mxArray * fbatch = mxGetField(options_mx,0, "fbatch"))
        options.opts.fbatch = (size_t)mxGetScalar(fbatch);
    if(mxArray * theta = mxGetField(options_mx,0, "theta"))
        options.opts.theta = mxGetScalar(theta);
    if(mxArray * iter = mxGetField(options_mx,0, "iter"))
        options.opts.iter = (size_t)mxGetScalar(iter);
    if(mxArray * verbose = mxGetField(options_mx,0, "verbose"))
        options.opts.verbose = (unsigned int)mxGetScalar(verbose);
    if(mxArray * nthreads = mxGetField(options_mx,0, "nthreads"))
        options.opts.nthreads = (int)mxGetScalar(nthreads);
    if(mxArray * extended = mxGetField(options_mx,0,"extended"))
        options.opts.extended = (bool)mxGetScalar(extended);
    if(mxArray * tol = mxGetField(options_mx, 0, "tol"))
        options.opts.tol = mxGetScalar(tol);
}

void printErrorExit(std::string const & str){
    std::cout << str << std::endl;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    neo_ica_options_type options;
    options.opts= neo_ica::options();

    mstream mout;
    std::streambuf* oldbuf = std::cout.rdbuf(&mout);

    if(nlhs!=2)
        return printErrorExit(USAGE_STR);

    //Check inputs and Outputs
    if(nrhs==1){
        if(!(mxIsSingle(prhs[0]) || mxIsDouble(prhs[0])))
            return printErrorExit("Invalid input arguments : The data must be a valid single/double matrix");
    }
    else if(nrhs==2){
        if(!mxIsStruct(prhs[1]))
            return printErrorExit("Invalid input arguments : The options must be a valid struct");
        mxArray * options_mx = mxDuplicateArray(prhs[1]);
        fill_options(options_mx, options);
    }
    else
        return printErrorExit(USAGE_STR);

    //Get dimensions
    const mwSize * dims = mxGetDimensions(prhs[0]);
    size_t NC = static_cast<size_t>(dims[0]);
    size_t NF = static_cast<size_t>(dims[1]);

    double * weights = NULL;
    plhs[0] = mxCreateDoubleMatrix(NC,NC,mxREAL);
    weights = mxGetPr(plhs[0]);

    double * sphere = NULL;
    plhs[1] = mxCreateDoubleMatrix(NC,NC,mxREAL);
    sphere = mxGetPr(plhs[1]);


    if(mxIsDouble(prhs[0])){
        //Get data
        double* data = mxGetPr(prhs[0]);
        transpose(data,NC,NF);

        neo_ica::ica(data, weights, sphere, NC, NF, options.opts);

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

        neo_ica::ica(data, weights_float, sphere_float, NC, NF, options.opts);

        for(size_t i = 0 ; i < NC ; ++i){
            for(size_t j = 0 ; j < NC ; ++j){
                weights[i*NC+j] = weights_float[j*NC+i];
                sphere[i*NC+j] = sphere_float[j*NC+i];
            }
        }

        transpose(data,NF,NC);
        delete[] weights_float;
        delete[] sphere_float;
    }

    std::cout.rdbuf(oldbuf);
}
