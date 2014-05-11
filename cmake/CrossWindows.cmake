macro(cross_windows COMPILER_PREFIX CROSS_PATH)

set(CMAKE_FIND_LIBRARY_SUFFIXES .dll .dll.a .a)

# the name of the target operating system
set(CMAKE_SYSTEM_NAME Windows)

set(CMAKE_EXECUTABLE_SUFFIX .exe)
# Choose an appropriate compiler prefix
# set the target environment location

set(CMAKE_PREFIX_PATH ${CROSS_PATH})
set(CMAKE_FIND_ROOT_PATH ${CROSS_PATH})

# which compilers to use for C and C++
set(CMAKE_RANLIB ${COMPILER_PREFIX}-ranlib)
set(CMAKE_AR ${COMPILER_PREFIX}-ar)
set(CMAKE_C_COMPILER ${COMPILER_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${COMPILER_PREFIX}-g++)
set(CMAKE_RC_COMPILER ${COMPILER_PREFIX}-windres)

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# disable -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

endmacro()
