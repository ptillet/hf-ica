# We need the mex compiler for this to work.
find_package(MatlabMex)

if(MATLAB_MEX_FOUND)

include_directories(${MATLAB_MEX_INCLUDE_DIRS})

get_property(DSHFICA_LOCATION TARGET dshf_ica PROPERTY LOCATION)
#set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "${DSHFICA_LOCATION}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fexceptions -fPIC -fno-omit-frame-pointer -pthread")
#get_property(DSHF_ICA_LOCATION TARGET dshf_ica PROPERTY LOCATION)

# Don't make 'lib<method>.mexglx'.
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_MODULE_PREFIX "")

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexa64")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".mexa64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86" OR ${CMAKE_SYSTEM_PROCESSOR}
    STREQUAL "i686")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexglx")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".mexglx")
endif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

# Set MATLAB toolbox install directory.
set(MATLAB_TOOLBOX_DIR "${MATLAB_ROOT}/toolbox")

set(CMAKE_CXX_CREATE_SHARED_MODULE "<CMAKE_CXX_COMPILER> -O -pthread -shared -Wl,--version-script,/opt/matlab/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined <LINK_FLAGS> -o <TARGET> <OBJECTS>  -Wl,-rpath-link,/opt/matlab/bin/glnxa64 -L/opt/matlab/bin/glnxa64 -lmx -lmex -lmat -lm <LINK_LIBRARIES>")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_MODULE}")

add_subdirectory(functions)

endif(MATLAB_MEX_FOUND)
