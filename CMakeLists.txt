project(ParICA)

cmake_minimum_required(VERSION 2.8)

set (PARICA_VERSION_MAJOR 0)
set (PARICA_VERSION_MINOR 0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(OpenCL QUIET REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
include_directories(".")

INCLUDE(CTest)
ENABLE_TESTING()
add_subdirectory(tests)

#QT Creator Hackery
file( GLOB_RECURSE QtCreatorCompatibility_SRC
    *.hpp *.h *.cl)
add_custom_target( QtCreatorCompatibility SOURCES ${QtCreatorCompatibility_SRC} )

set(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(SRC ${SRC_PATH}/linear_ica.cpp)
add_library(clica ${SRC})
set_target_properties(clica PROPERTIES COMPILE_FLAGS "-fPIC")

#OpenMP
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

add_subdirectory(examples)
add_subdirectory(matlab)

set(CMAKE_BUILD_TYPE release)
