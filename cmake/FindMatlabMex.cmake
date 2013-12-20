SET(MATLAB_MEX_FOUND 0)

SET(MATLAB_MEX_INCLUDE_DIR_PATHS      ${MATLAB_ROOT}/extern/include
                              /opt/matlab/extern/include
                              /usr/local/matlab/extern/include
                              $ENV{HOME}/matlab/extern/include
                              # Now all the versions
                              /opt/matlab/[rR]20[0-9][0-9][abAB]/extern/include
                              /usr/local/matlab/[rR]20[0-9][0-9][abAB]/extern/include
                              /opt/matlab-[rR]20[0-9][0-9][abAB]/extern/include
                              /opt/matlab_[rR]20[0-9][0-9][abAB]/extern/include
                              /usr/local/matlab-[rR]20[0-9][0-9][abAB]/extern/include
                              /usr/local/matlab_[rR]20[0-9][0-9][abAB]/extern/include
                              $ENV{HOME}/matlab/[rR]20[0-9][0-9][abAB]/extern/include
                              $ENV{HOME}/matlab-[rR]20[0-9][0-9][abAB]/extern/include
                              $ENV{HOME}/matlab_[rR]20[0-9][0-9][abAB]/extern/include)

SET(MATLAB_MEX_LIBRARY_PATHS   ${MATLAB_ROOT}/bin/glnxa64
                              /opt/matlab/bin/glnxa64
                              /usr/local/matlab/bin/glnxa64
                              $ENV{HOME}/matlab/bin/glnxa64
                              # Now all the versions
                              /opt/matlab/[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              /usr/local/matlab/[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              /opt/matlab-[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              /opt/matlab_[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              /usr/local/matlab-[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              /usr/local/matlab_[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              $ENV{HOME}/matlab/[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              $ENV{HOME}/matlab-[rR]20[0-9][0-9][abAB]/bin/glnxa64
                              $ENV{HOME}/matlab_[rR]20[0-9][0-9][abAB]/bin/glnxa64)

find_path(MATLAB_MEX_INCLUDE_DIR NAMES mex.h PATHS ${MATLAB_MEX_INCLUDE_DIR_PATHS})
find_library(MATLAB_LIBMEX NAMES mex PATHS ${MATLAB_MEX_LIBRARY_PATHS})
find_library(MATLAB_LIBMAT NAMES mat PATHS ${MATLAB_MEX_LIBRARY_PATHS})
find_library(MATLAB_LIBMENG NAMES eng PATHS ${MATLAB_MEX_LIBRARY_PATHS})
find_library(MATLAB_LIBMX NAMES mx PATHS ${MATLAB_MEX_LIBRARY_PATHS})


mark_as_advanced(MATLAB_MEX_LIBRARY MATLAB_MEX_INCLUDE_DIR)

set(MATLAB_MEX_INCLUDE_DIRS ${MATLAB_MEX_INCLUDE_DIR})
set(MATLAB_MEX_LIBRARIES ${MATLAB_LIBMEX} ${MATLAB_LIBMAT} ${MATLAB_LIBMENG} ${MATLAB_LIBMX})


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MATLAB_MEX  DEFAULT_MSG
                                  MATLAB_MEX_LIBRARIES MATLAB_MEX_INCLUDE_DIR)
