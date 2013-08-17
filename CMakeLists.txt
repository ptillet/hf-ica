project(CLICA)

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
set(SRC ${SRC_PATH}/linear_ica.cpp
        ${SRC_PATH}/whiten.cpp)
add_library(clica ${SRC})
set_target_properties(clica PROPERTIES COMPILE_FLAGS "-DFMINCL_WITH_EIGEN")

add_subdirectory(examples)
add_subdirectory(matlab)

set(CMAKE_BUILD_TYPE release)
