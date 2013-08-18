#include <math.h>
#include <matrix.h>
#include <mex.h>

#include "parica.h"
#include <Eigen/Dense>

typedef double NumericT;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nlhs>1)
        mexErrMsgTxt("Signature : result (data)");
    if(nrhs>1)
        mexErrMsgTxt("Signature : result (data)");

    //Get data
    mxArray * data_tmp = mxDuplicateArray(prhs[0]);
    NumericT * data = mxGetPr(data_tmp);

    //Get dimensions
    const mwSize * dims = mxGetDimensions(prhs[0]);
    std::size_t size1 = static_cast<std::size_t>(dims[0]);
    std::size_t size2 = static_cast<std::size_t>(dims[1]);

    mxArray* result_tmp = plhs[0] = mxCreateDoubleMatrix(size1,size2,mxREAL);
    NumericT * result = mxGetPr(result_tmp);

    Eigen::Map<Eigen::MatrixXd> map_data(data,size1,size2);
    Eigen::Map<Eigen::MatrixXd> map_result(result,size1,size2);

    parica::inplace_linear_ica(map_data, map_result);
}
