add_executable(artificial_sources artificial_sources.cpp ${PROJECT_SOURCE_DIR}/src/whiten.cpp ${PROJECT_SOURCE_DIR}/src/linear_ica.cpp)
target_link_libraries(artificial_sources ${OPENCL_LIBRARIES})
set_target_properties(artificial_sources PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
