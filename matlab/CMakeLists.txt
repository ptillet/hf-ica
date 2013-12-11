set(MATLAB /opt/matlab)
set(MEX_EXT .mexa64)

#### define compile switches
set(MATLAB_CFLAGS "-fexceptions -fPIC -fno-omit-frame-pointer -pthread")

set(CMAKE_C_FLAGS_DEBUG    "-g -Wall")
set(CMAKE_CXX_FLAGS_DEBUG  "-g -Wall")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -funroll-all-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-all-loops")

#### platform specific defines
if(UNIX)
  set(PLATFORM "_UNIX_")
  set(MEXDEFFILE "")
endif(UNIX)
if(WIN32)
  set(PLATFORM "_WIN32_")
  set(MEXDEFFILE ../../MATLAB/mexfunction.def)
  set(MEXLIB "libmx;libmex")
endif(WIN32)
if(APPLE)
  set(PLATFORM "_UNIX_")
  set(MEXDEFFILE "")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined
dynamic_lookup")
endif(APPLE)

#### common defines
set(CDEFS "-D${PLATFORM}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MATLAB_CFLAGS} ${CDEFS} ${OpenMP_C_FLAGS} -largeArrayDims")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${MATLAB_CFLAGS} ${CDEFS} ${OpenMP_CXX_FLAGS} -largeArrayDims ")
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")

#### define target-specific flags
set(BUILD_SHARED_LIBS 1)
if(WIN32)
  add_definitions(-D_MAKEDLL_)
  link_directories("${MATLAB}/extern/lib/win32/microsoft")
  add_definitions(-D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1)
  message(STATUS "- MSVC: _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1")
  add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
  message(STATUS "- MSVC: _CRT_NONSTDC_NO_WARNINGS")
endif(WIN32)

#### add subdirectories containing include files (MEX)
include_directories("${MATLAB}/extern/include")
include_directories("${MATLAB}/simulink/include")

# Set MATLAB toolbox install directory.
set(MATLAB_TOOLBOX_DIR "${MATLAB}/toolbox")

add_subdirectory(functions)
