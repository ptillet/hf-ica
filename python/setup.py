#Thanks to Andreas Knoeckler for providing stand-alone boost.python
#through PyOpenCL and PyCUDA

import os, sys
from distutils.ccompiler import show_compilers,new_compiler
from distutils.command.build_ext import build_ext
from distutils.command.build_py import build_py
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from distutils import sysconfig
from imp import find_module
from glob import glob
from os.path import dirname
import numpy as np

platform_cflags = {}
platform_ldflags = {}
platform_libs = {}

class build_ext_subclass(build_ext):
    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

def recursive_glob(rootdir='.', suffix=''):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

def main():
    
    #Numpy
    include = [os.path.join(find_module('numpy')[1], 'core', 'include')]
    src = []
    
    #Neo-ica
    include += [os.path.join('src', 'include')]
    src +=  recursive_glob(os.path.join('src','lib'), 'cpp')
    
    #Bindings
    include += [os.path.join('src', 'bind')]
    src += recursive_glob(os.path.join('src','bind'), 'cpp')

    #BLAS/Lapack
    blas = np.__config__.blas_opt_info
    lapack = np.__config__.lapack_opt_info
    
    #Extensions
    libraries = blas['libraries'] + lapack['libraries']
    library_dirs = blas['library_dirs'] + lapack['library_dirs']
    lib = Extension('_ica',
                    sources=src,
                    libraries=libraries,
                    library_dirs=library_dirs,
                    extra_compile_args=['-std=c++11', '-fopenmp', '-msse4'],
                    extra_link_args=['-lgomp', '-Wl,-soname=_ica.so'],
                    include_dirs=include)
    
    #Setup
    setup(
          name='neo_ica',
          version='1.0',
          description="NEO-ICA",
          author='Philippe Tillet',
          author_email='ptillet@g.harvard.edu',
          license='MIT',
          packages=['neo_ica'],
          ext_package='neo_ica',
          ext_modules=[lib],
          cmdclass={'build_py': build_py, 'build_ext': build_ext_subclass},
          classifiers=['Environment :: Console',
                       'Development Status :: 4 - Beta',
                       'Intended Audience :: Developers',
                       'Intended Audience :: Other Audience',
                       'Intended Audience :: Science/Research',
                       'License :: OSI Approved :: MIT License',
                       'Natural Language :: English',
                       'Programming Language :: C++',
                       'Programming Language :: Python',
                       'Programming Language :: Python :: 3',
                       'Topic :: Scientific/Engineering',
                       'Topic :: Scientific/Engineering :: Mathematics',
                       'Topic :: Scientific/Engineering :: Signal Processing',
                       'Topic :: Scientific/Engineering :: Physics',
                       'Topic :: Scientific/Engineering :: Machine Learning']
         )

if __name__ == "__main__":
    main()
