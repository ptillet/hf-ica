project(CLICA)

cmake_minimum_required(VERSION 2.8)

set (PARICA_VERSION_MAJOR 0)
set (PARICA_VERSION_MINOR 0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(OpenCL QUIET REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

#QT Creator Hackery
file( GLOB_RECURSE QtCreatorCompatibility_SRC
    *.hpp *.h *.cl)
add_custom_target( QtCreatorCompatibility SOURCES ${QtCreatorCompatibility_SRC} )


INCLUDE_DIRECTORIES(".")

INCLUDE(CTest)
ENABLE_TESTING()
add_subdirectory(tests)

add_subdirectory(examples)

