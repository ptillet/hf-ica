project(ParICA)

cmake_minimum_required(VERSION 2.8)

set (PARICA_VERSION_MAJOR 0)
set (PARICA_VERSION_MINOR 0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(".")

INCLUDE(CTest)
ENABLE_TESTING()
add_subdirectory(tests)

#QT Creator Hackery
file( GLOB_RECURSE QtCreatorCompatibility_SRC
    *.hpp *.h *.cl)


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_BUILD_TYPE release)

##OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
endif()

add_custom_target( QtCreatorCompatibility SOURCES ${QtCreatorCompatibility_SRC} )
find_library(OPENBLAS_LIB libopenblas)
set(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(SRC ${SRC_PATH}/linear_ica.cpp)
add_library(clica ${SRC})
set_target_properties(clica PROPERTIES COMPILE_FLAGS "-fPIC")
target_link_libraries(clica openblas.a gfortran)



add_subdirectory(examples)
add_subdirectory(matlab)

