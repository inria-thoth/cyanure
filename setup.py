from setuptools import setup, Extension, find_packages
import numpy
import platform
import struct

import contextlib
import os

# Override sdist to always produce .zip archive
from distutils.command.sdist import sdist as _sdist

class sdistzip(_sdist):
    def initialize_options(self):
        _sdist.initialize_options(self)
        self.formats = ['zip', 'gztar']
    
def getBlas():
    file_ = open("npConfg_file.txt", "w")
    with contextlib.redirect_stdout(file_):
        numpy.show_config()
    file_.close()
    np_confg = open('npConfg_file.txt', 'r')
    lib = ""
    for line in np_confg:
        if 'libraries' in line:
            lib = line
            break
    np_confg.close()
    os.remove("npConfg_file.txt")
    if lib != "":
        blas = lib.split('[')[1].split(',')[0]
        return blas[1:len(blas) - 1]
    else:
        return lib


np_blas = getBlas()

LIBS = []
INCLUDE_DIRS = []
EXTRA_COMPILE_ARGS = []
LIBRARY_DIRS = []
RUNTIME_LIRABRY_DIRS = []

if platform.system() == "Windows":
    if 'mkl' in np_blas:
        libs = ['mkl_rt', 'iomp5']
        extra_compile_args = [
            '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '/permissive-', '/W1']

    else: 
        extra_compile_args = [
            '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '/PIC',
            '/permissive-', '/W1']
        libs = ['lapack', 'blas']
    
    LIBS = libs
    INCLUDE_DIRS = [numpy.get_include()]
    EXTRA_COMPILE_ARGS = extra_compile_args

    if struct.calcsize("P") * 8 == 32:
        INCLUDE_DIRS = ['D:/a/cyanure/cyanure/openblas_86/include'] + INCLUDE_DIRS
        LIBRARY_DIRS = ['D:/a/cyanure/cyanure/openblas_86/lib', 'D:/a/cyanure/cyanure/lapack']
    else:
        INCLUDE_DIRS = ['D:/a/cyanure/cyanure/openblas_64/include'] + INCLUDE_DIRS
        LIBRARY_DIRS = ['D:/a/cyanure/cyanure/openblas_64/lib', 'D:/a/cyanure/cyanure/lapack']

else:
    ##### setup mkl_rt
    if 'mkl' in np_blas:
        extra_compile_args_mkl = [
            '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '-fPIC',
            '-fopenmp', '-std=c++11']

        LIBS = ['mkl_rt', 'iomp5']

        INCLUDE_DIRS = [numpy.get_include()]
        EXTRA_COMPILE_ARGS = extra_compile_args_mkl

    ##### setup openblas
    else:
    
        libs = ['lapack', 'blas']
        
        extra_compile_args = [
            '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC',
            '-std=c++11', '-fopenmp']      

        INCLUDE_DIRS = [numpy.get_include()]

        INCLUDE_DIRS = ['/usr/local/opt/openblas/include'] + INCLUDE_DIRS
        LIBRARY_DIRS = ['/usr/local/opt/openblas/lib']
        LIBS = libs
        RUNTIME_LIRABRY_DIRS = LIBRARY_DIRS
        EXTRA_COMPILE_ARGS = extra_compile_args

        if platform.system() == "Darwin":
            INCLUDE_DIRS = ["/usr/local/include", "/usr/local/opt/llvm/include"] + INCLUDE_DIRS
            LIBRARY_DIRS = ["/usr/local/lib", "/usr/local/opt/llvm/lib"] + LIBRARY_DIRS

    if "COVERAGE" in os.environ:
        EXTRA_COMPILE_ARGS = EXTRA_COMPILE_ARGS + ['-fprofile-arcs', '-ftest-coverage']
        LIBS = LIBS + ['gcov'] 


if platform.system() != "Windows":
    EXTRA_LINK_ARGS = ['-fopenmp']
    if "COVERAGE" in os.environ:
        EXTRA_LINK_ARGS = EXTRA_LINK_ARGS + ['-fprofile-arcs']
else:
    EXTRA_LINK_ARGS = []
   

cyanure_wrap = Extension(
    'cyanure_lib.cyanure_wrap',
    libraries=LIBS,
    include_dirs=INCLUDE_DIRS,
    language='c++',
    library_dirs=LIBRARY_DIRS,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    runtime_library_dirs=RUNTIME_LIRABRY_DIRS,
    extra_link_args=EXTRA_LINK_ARGS,
    sources=['cyanure_lib/cyanure_wrap_module.cpp'])

setup(name='cyanure',
      version='0.22.4',
      author="Julien Mairal",
      author_email="julien.mairal@inria.fr",
      license='bsd-3-clause',
      url="http://julien.mairal.org/cyanure/",
      description='optimization toolbox for machine learning',
      install_requires=['scipy', 'numpy>=1.18.0', 'scikit-learn'],
      ext_modules=[cyanure_wrap],
      packages=find_packages(),
      cmdclass={'sdist': sdistzip},
      py_modules=['cyanure'])
