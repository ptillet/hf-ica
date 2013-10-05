foreach(F curveica)
        #add_library(${F}_mex SHARED ${F}.cpp ${SRC_PATH}/linear_ica.cpp)
        add_library(${F}_mex SHARED ${F}.cpp)
        target_link_libraries(${F}_mex curveica)
        set_target_properties(${F}_mex PROPERTIES OUTPUT_NAME ${F})
# Installation rule.  Install both the mex and the MATLAB file.
        install(TARGETS ${F}_mex  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/curveica/")
endforeach()
