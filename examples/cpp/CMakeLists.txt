foreach(PROG artificial_sources)
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} neo_ica ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endforeach(PROG)
