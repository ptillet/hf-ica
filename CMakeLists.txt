project(DSHF-ICA)

cmake_minimum_required(VERSION 2.8)

set(DSHF-ICA_VERSION_MAJOR 1)
set(DSHF-ICA_VERSION_MINOR 0)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

#Cross-Compile
option(CROSS_WIN64 "Cross compile for Win64 using MinGW32" OFF)
if(CROSS_WIN64)
    include(CrossMinGW32)
endif()

#Modules

#Include
include_directories(".")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OPENBLAS_INCLUDE_DIR})

#Tests
include(CTest)
enable_testing()
add_subdirectory(tests)


#QT Creator Hackery
file( GLOB_RECURSE QtCreatorCompatibility_SRC
    *.hpp *.h *.cl)
add_custom_target( QtCreatorCompatibility SOURCES ${QtCreatorCompatibility_SRC} )


set(CMAKE_CXX_FLAGS_RELEASE "-funroll-loops -O3")
set(CMAKE_BUILD_TYPE Release)

#OpenBlas
find_package(OpenBlas)

##OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
endif()



set(SRC_PATH "${PROJECT_SOURCE_DIR}/src")
set(SRC ${SRC_PATH}/linear_ica.cpp
        ${SRC_PATH}/nonlinearities/extended_infomax.cpp)
add_library(dshf_ica ${SRC})

IF( ${CMAKE_COMPILER_IS_GNUCXX} )
  SET(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -static-libstdc++ -static-libgcc")
ENDIF( ${CMAKE_COMPILER_IS_GNUCXX} )

set_target_properties(dshf_ica PROPERTIES COMPILE_FLAGS "-msse3 -fPIC")

target_link_libraries(dshf_ica ${OPENBLAS_LIB})



add_subdirectory(examples)
add_subdirectory(matlab)

