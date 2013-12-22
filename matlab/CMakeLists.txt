# We need the mex compiler for this to work.
find_package(MatlabMex)

if(MATLAB_MEX_FOUND)

include_directories(${MATLAB_MEX_INCLUDE_DIRS})

get_property(DSHFICA_LOCATION TARGET dshf_ica PROPERTY LOCATION)
#set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "${DSHFICA_LOCATION}")
#get_property(DSHF_ICA_LOCATION TARGET dshf_ica PROPERTY LOCATION)

# Don't make 'lib<method>.mexglx'.
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_MODULE_PREFIX "")

if(_WIN64)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexw64")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".mexw64")
elseif(_WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexw32")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".mexw32")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
      set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexw64")
      set(CMAKE_SHARED_MODULE_SUFFIX  ".mexw64")
   else()
      set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexa64")
      set(CMAKE_SHARED_MODULE_SUFFIX  ".mexa64")
   endif()
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86" OR ${CMAKE_SYSTEM_PROCESSOR}
    STREQUAL "i686")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexglx")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".mexglx")
endif()

# Set MATLAB toolbox install directory.
set(MATLAB_TOOLBOX_DIR "${MATLAB_ROOT}/toolbox")

if(_WIN64 OR _WIN32)
    set(MEX_ARCHITECTURE_OPTIONS " -fPIC --export-all-symbols")
else()
    set(MEX_ARCHITECTURE_OPTIONS "-pthread -Wl,--version-script,${MATLAB_ROOT}/extern/lib/${MATLAB_ARCH}/mexFunction.map")
endif()

set(CMAKE_CXX_CREATE_SHARED_MODULE "<CMAKE_CXX_COMPILER> -O -shared -fexceptions -fno-omit-frame-pointer ${MEX_ARCHITECTURE_OPTIONS} -Wl,--no-undefined <LINK_FLAGS> -o <TARGET> <OBJECTS>  -Wl,-rpath-link,${MATLAB_MEX_LIBRARY_PATHS} -L${MATLAB_MEX_LIBRARY_PATHS} -lmx -lmex -lmat -lm <LINK_LIBRARIES>")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_MODULE}")

add_subdirectory(functions)

endif(MATLAB_MEX_FOUND)
