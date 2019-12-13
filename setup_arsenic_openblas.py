from distutils.core import setup, Extension
import numpy

libs = ['openblas','gomp']
libs = ['openblas']

arsenic_wrap=Extension('arsenic_wrap', 
        libraries = libs,
        include_dirs=[numpy.get_include()],
        language='c++',
        extra_compile_args = ['-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC', '-std=c++11', '-Wno-unused-function', '-Wno-write-strings' , '-fmax-errors=5'],
        #extra_compile_args = ['-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC', '-fopenmp', '-std=c++11', '-Wno-unused-function', '-Wno-write-strings' , '-fmax-errors=5'],
        sources=['arsenic_wrap_module.cpp'])

setup(name='arsenic',
        version='1.0',
        description='optimization toolbox',
        ext_modules=[arsenic_wrap],
        py_modules=['arsenic'])


