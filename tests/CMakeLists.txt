 add_executable(whiten whiten.cpp ${PROJECT_SOURCE_DIR}/src/whiten.cpp)
 target_link_libraries(whiten ${OPENCL_LIBRARIES})
 set_target_properties(whiten PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
