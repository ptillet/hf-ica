project(neo_ica)

cmake_minimum_required(VERSION 2.8)

set(NEO_ICA_VERSION_MAJOR 1)
set(NEO_ICA_VERSION_MINOR 0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

#Make all files visible to some IDEs
file( GLOB_RECURSE QtCreatorCompatibility_SRC
    *.hpp *.h *.cpp)
add_custom_target( QtCreatorCompatibility SOURCES ${QtCreatorCompatibility_SRC} )

#Cross-Compile
option(BUILD_EXAMPLE "Build simple C++ example" OFF)
option(MEX "Build MATLAB MEX file" ON)

#Matlab ROOT
set(MATLAB_ROOT "" CACHE STRING "Root of the MATLAB environment")

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
set(CMAKE_CXX_FLAGS_RELEASE "-funroll-loops -O2")
set(CMAKE_BUILD_TYPE Release)

##OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
endif()

##Sources
set(NEO_ICA_SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(NEO_ICA_SRC ${NEO_ICA_SRC_PATH}/ica.cpp
        ${NEO_ICA_SRC_PATH}/dist/sejnowski.cpp)

##Adds static library and examples
if(NOT _WIN32 AND NOT _WIN64)
    add_library(neo_ica ${NEO_ICA_SRC})
    set_target_properties(neo_ica PROPERTIES COMPILE_FLAGS "-fPIC")
    if(BUILD_EXAMPLE)
        add_subdirectory(examples)
    endif()
endif()

##MATLAB
find_package(MatlabMex)
if(MATLAB_MEX_FOUND)
    add_subdirectory(matlab)
endif()
