project(DSHF-ICA)

cmake_minimum_required(VERSION 2.8)

set(DSHF-ICA_VERSION_MAJOR 1)
set(DSHF-ICA_VERSION_MINOR 0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

#Cross-Compile
include(CrossWindows)

option(CROSS_WIN32 "Cross compile for Win32 using MinGW32" OFF)
option(CROSS_WIN64 "Cross compile for Win64 using MinGW32" OFF)
option(MEX "Build MATLAB MEX file" ON)

set(MATLAB_ROOT "" CACHE STRING "Root of the MATLAB environment")

if(CROSS_WIN64)
    cross_windows(x86_64-w64-mingw32 /opt/cross_win64/)
elseif(CROSS_WIN32)
    cross_windows(i686-w64-mingw32 /opt/cross_win32/)
endif()

set(_WIN64 ${CROSS_WIN64})
set(_WIN32 ${CROSS_WIN32})

if(MEX)
    if(_WIN64)
        set(MATLAB_ARCH win64)
    elseif(_WIN32)
        set(MATLAB_ARCH win32)
    else()
        set(MATLAB_ARCH glnxa64)
    endif()
endif()

#Include
include_directories(".")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_PREFIX_PATH}/include)
include_directories(${OPENBLAS_INCLUDE_DIR})

#Tests
include(CTest)
enable_testing()
add_subdirectory(tests)


#QT Creator Hackery
file( GLOB_RECURSE QtCreatorCompatibility_SRC
    *.hpp *.h *.cl)
add_custom_target( QtCreatorCompatibility SOURCES ${QtCreatorCompatibility_SRC} )

set(CMAKE_CXX_FLAGS_RELEASE "-funroll-loops -O3 -msse3")
set(CMAKE_BUILD_TYPE Release)

#OpenBlas
#find_package(OpenBlas)

##OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
endif()



set(DSHF_ICA_SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(DSHF_ICA_SRC ${DSHF_ICA_SRC_PATH}/linear_ica.cpp
        ${DSHF_ICA_SRC_PATH}/nonlinearities/extended_infomax.cpp)
add_library(dshf_ica ${DSHF_ICA_SRC})

#set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -static-libgcc -static-libstdc++")

set_target_properties(dshf_ica PROPERTIES COMPILE_FLAGS "-fPIC")
target_link_libraries(dshf_ica openblas.a gfortran)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()


add_subdirectory(examples)
add_subdirectory(matlab)

