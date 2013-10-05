foreach(PROG artificial_sources)
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} curveica)
endforeach(PROG)
