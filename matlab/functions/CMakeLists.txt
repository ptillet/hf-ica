foreach(F linear_parica)
	add_library(${F}_mex SHARED ${F}.cpp)
	target_link_libraries(${F}_mex clica ${OPENCL_LIBRARIES})
	set_target_properties(${F}_mex PROPERTIES OUTPUT_NAME ${F})
# Installation rule.  Install both the mex and the MATLAB file.
        install(TARGETS ${F}_mex  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/parica/")
endforeach()
