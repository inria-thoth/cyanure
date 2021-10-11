from setuptools import setup, Extension
import numpy
import platform
import sys 

import contextlib
import os

if platform.system() == "Darwin":
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"

def getBlas():
    file_ = open("npConfg_file.txt","w")
    with contextlib.redirect_stdout(file_):
        numpy.show_config()
    file_.close()
    np_confg = open('npConfg_file.txt','r')
    for line in np_confg:
        if 'libraries' in line:
            lib = line
            break
    np_confg.close()
    blas = lib.split('[')[1].split(',')[0]
    os.remove("npConfg_file.txt")
    return blas[1:len(blas)-1]

np_blas = getBlas()

LIBS = []
INCLUDE_DIRS = []
EXTRA_COMPILE_ARGS = []

if platform.system() == "Windows":
    libs_mkl_windows = ['mkl_rt', 'iomp5']
    include_dirs_mkl_windows = [numpy.get_include()]
    extra_compile_args_mkl_windows = [
            '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '/permissive-', '/W1']
    LIBS = libs_mkl_windows
    INCLUDE_DIRS = include_dirs_mkl_windows
    EXTRA_COMPILE_ARGS = extra_compile_args_mkl_windows

else:
    ##### setup mkl_rt
    if 'mkl' in np_blas:
        extra_compile_args_mkl = [
                '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '-fPIC',
                '-fopenmp', '-std=c++11']
        libs_mkl = ['mkl_rt', 'iomp5']
        include_dirs_mkl = [numpy.get_include()]

        LIBS = libs_mkl
        INCLUDE_DIRS = include_dirs_mkl
        EXTRA_COMPILE_ARGS = extra_compile_args_mkl

    ##### setup openblas
    if 'blas' in np_blas:
        extra_compile_args_open_blas=[
                '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC', '-fopenmp',
                '-std=c++11']
        libs_open_blas = [np_blas]
        include_dirs_open_blas = [numpy.get_include(), '/usr/local/lib/']

        LIBS = libs_open_blas
        INCLUDE_DIRS = include_dirs_open_blas
        EXTRA_COMPILE_ARGS = extra_compile_args_open_blas

        if platform.system() == "Darwin":            
            os.system("find /usr/local -xdev -name '*libopenblas*'")
            INCLUDE_DIRS = ['/usr/local/opt/openblas/include', '/usr/local/opt/llvm/include'] + INCLUDE_DIRS
            LIBS = LIBS + ['libomp']

print("DEBUG INSTALL: " + np_blas)

"""
## setup openblass no openmp

libs_open_blass_no_openmp = ['openblas']
include_dirs_open_blass_no_openmp = [numpy.get_include()]
extra_compile_args_open_blass_no_openmp =[
            '-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC', '-std=c++11']

#### setup mkl no openmp
libs_mkl_no_openmp = ['mkl_rt']
include_dirs_mkl_no_openmp = [numpy.get_include()]
extra_compile_args_mkl_no_openmp =[
                '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '-fPIC',
                '-std=c++11']
n argumentss

"""

cyanure_wrap = Extension(
    'cyanure_wrap',
    libraries=LIBS,
    include_dirs=INCLUDE_DIRS,
    language='c++',
    extra_compile_args=EXTRA_COMPILE_ARGS,
    sources=['cyanure_wrap_module.cpp'])

setup(name='cyanure setup',
      version='0.22post2',
      author="Julien Mairal",
      author_email="julien.mairal@inria.fr",
      license='bsd-3-clause',
      url="http://julien.mairal.org/cyanure/",
      description='optimization toolbox for machine learning',
      install_requires=['scipy', 'numpy'],
      ext_modules=[cyanure_wrap],
      py_modules=['cyanure'])

