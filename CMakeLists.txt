project(neo_ica)

cmake_minimum_required(VERSION 2.8)

set(NEO_ICA_VERSION_MAJOR 1)
set(NEO_ICA_VERSION_MINOR 0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

#Make all files visible to some IDEs
file( GLOB_RECURSE QtCreatorCompatibility_SRC
    *.hpp *.h *.cpp)
add_custom_target( QtCreatorCompatibility SOURCES ${QtCreatorCompatibility_SRC} )

#OpenBlas
#find_package(OpenBlas)

#Include
include_directories(".")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_PREFIX_PATH}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/include)

#Tests
include(CTest)
enable_testing()
add_subdirectory(tests)

#FLAGS
if(MSVC)
    set(VECTOR_FLAG "/arch:SSE2")
    set(WARNING_FLAG "/W2")
else()
    set(VECTOR_FLAG "-msse4")
    set(WARNING_FLAG "-Wall -Wextra")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAG} ${VECTOR_FLAG}")
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

#Default to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
endif()

#Sources
set(NEO_ICA_SRC_PATH "${PROJECT_SOURCE_DIR}/src")
file(GLOB_RECURSE NEO_ICA_SRC ${NEO_ICA_SRC_PATH}/*.cpp)

#Library
add_library(neo_ica ${NEO_ICA_SRC})
if(NOT WIN32)
    set_target_properties(neo_ica PROPERTIES COMPILE_FLAGS "-fPIC")
endif()

#Examples
add_subdirectory(examples)

#MATLAB
set(MATLAB_ROOT "" CACHE STRING "Root of the MATLAB environment")
find_package(MatlabMex)
if(MATLAB_MEX_FOUND)
    add_subdirectory(matlab)
endif()
