#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <cstring>
#include "parica.h"
#include <Eigen/Dense>

typedef double NumericT;

inline bool are_string_equal(const char * a, const char * b){
    return std::strcmp(a,b)==0;
}

struct parica_options_type{
    bool use_float;
    fmincl::optimization_options optimization;
};


void fill_options(mxArray* options_mx, parica_options_type & options){

    /*-Max Iter-*/
    if(mxArray * use_float = mxGetField(options_mx,0, "useFloat")){
        options.use_float = static_cast<bool>(mxGetScalar(use_float));
    }

    fmincl::optimization_options & optimization = options.optimization;
    /*-Max Iter-*/
    if(mxArray * max_iter = mxGetField(options_mx,0, "maxIter"))
        optimization.max_iter = mxGetScalar(max_iter);

    /*-Verbosity Level-*/
    if(mxArray * verbosity_level = mxGetField(options_mx,0, "verbosityLevel")){
        optimization.verbosity_level = mxGetScalar(verbosity_level);
    }

    /*-Direction-*/
    if(mxArray * direction = mxGetField(options_mx,0,"direction")){
        char* direction_name = mxArrayToString(direction);

        //Quasi-Newton
        if(are_string_equal(direction_name,"qn")){
            fmincl::quasi_newton_tag * direction = new fmincl::quasi_newton_tag();

            //Quasi-Newton Update overriden
            if(mxArray * qn_update = mxGetField(options_mx,0,"qnUpdate")){
                char* qn_update_name = mxArrayToString(qn_update);

                //"LBFGS"
                if(are_string_equal(qn_update_name,"lbfgs")){
                    fmincl::lbfgs_tag * lbfgs = new fmincl::lbfgs_tag();
                    //Set LBFGS memory
                    if(mxArray * lbfgs_memory = mxGetField(options_mx,0,"lbfgsMemory")){
                        lbfgs->m = mxGetScalar(lbfgs_memory);
                    }
                    direction->update = lbfgs;
                }

                //"BFGS"
                else if(are_string_equal(qn_update_name,"bfgs")){
                    fmincl::qn_update_tag * bfgs = new fmincl::bfgs_tag();
                    direction->update = bfgs;
                }

                //"Invalid"
                else
                    mexErrMsgIdAndTxt( "parica:invalidQnUpdate",
                                       "Please specify a valid qnUpdate. Supported for now : \n \"lbfgs\", \"bfgs\" ");
            }
            optimization.direction = direction;
        }

        //Conjugate Gradients
        else if(are_string_equal(direction_name,"cg")){
            fmincl::cg_tag * direction = new fmincl::cg_tag();

            //CG Update overriden
            if(mxArray * cg_update = mxGetField(options_mx,0,"cgUpdate")){
                char * cg_update_name = mxArrayToString(cg_update);

                if(are_string_equal(cg_update_name,"PolakRibiere"))
                    direction->update = new fmincl::polak_ribiere_tag();

                else
                    mexErrMsgIdAndTxt( "parica:invalidCgUpdate",
                                       "Please specify a valid cgUpdate. Supported for now : \n \"PolakRibiere\" ");
            }

            //CG Restart overriden
            if(mxArray * cg_restart = mxGetField(options_mx,0,"cgRestart")){
                char * cg_restart_name = mxArrayToString(cg_restart);

                if(are_string_equal(cg_restart_name,"noRestart"))
                    direction->restart = new fmincl::no_restart_tag();

                else
                    mexErrMsgIdAndTxt( "parica:invalidCgRestart",
                                       "Please specify a valid cgRestart. Supported for now : \n \"noRestart\" ");
            }

            optimization.direction = direction;
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    parica_options_type options;
    //Set default
    options.use_float = true;
    options.optimization = parica::make_default_options();


    if(nrhs>2)
        mexErrMsgIdAndTxt( "parica:invalidNumInputs",
                           "Invalid input arguments : independent_components = linear_parica(data [, options])");
    if(nrhs>1){
        if(mxIsStruct(prhs[1])){
            mxArray * options_mx = mxDuplicateArray(prhs[1]);
            fill_options(options_mx, options);
        }
        else
            mexErrMsgIdAndTxt("parica:invalidOptionsType",
                              "Invalid input arguments : The options must be a valid struct");
    }
    if(nlhs>1)
        mexErrMsgIdAndTxt( "parica:TooManyOutputArguments",
                           "Too many output arguments : independent_components = linear_parica(data [, options])");
    //Get data
    mxArray * data_tmp = mxDuplicateArray(prhs[0]);
    NumericT * data = mxGetPr(data_tmp);

    //Get dimensions
    const mwSize * dims = mxGetDimensions(prhs[0]);
    std::size_t NC = static_cast<std::size_t>(dims[0]);
    std::size_t NF = static_cast<std::size_t>(dims[1]);

    mxArray* result_tmp = plhs[0] = mxCreateDoubleMatrix(NC,NF,mxREAL);
    NumericT * result = mxGetPr(result_tmp);

    Eigen::Map<Eigen::MatrixXd> map_data(data,NC,NF);
    Eigen::Map<Eigen::MatrixXd> map_result(result,NC,NF);
    if(options.use_float){
        parica::result_of::internal_matrix_type<float>::type data_eigen = map_data.cast<float>();
        parica::result_of::internal_matrix_type<float>::type result_eigen(NC,NF);
        parica::inplace_linear_ica(data_eigen.data(), result_eigen.data(),NC,NF,options.optimization);
        map_result = result_eigen.cast<double>();
    }
    else{
        parica::result_of::internal_matrix_type<double>::type data_eigen = map_data;
        parica::result_of::internal_matrix_type<double>::type result_eigen(NC,NF);
        parica::inplace_linear_ica(data_eigen.data(), result_eigen.data(),NC,NF,options.optimization);
        map_result = result_eigen;
    }
}
