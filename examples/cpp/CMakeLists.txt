foreach(PROG artificial_sources)
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} neo_ica openblas gfortran)
endforeach(PROG)
