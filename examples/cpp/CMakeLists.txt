foreach(PROG artificial_sources)
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} clica ${OPENCL_LIBRARIES})
endforeach(PROG)
