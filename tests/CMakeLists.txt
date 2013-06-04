add_executable(whiten-test whiten.cpp ${PROJECT_SOURCE_DIR}/src/whiten.cpp)
target_link_libraries(whiten-test ${OPENCL_LIBRARIES})
set_target_properties(whiten-test PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
add_test(whiten whiten-test)
