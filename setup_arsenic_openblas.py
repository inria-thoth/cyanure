from setuptools import setup, Extension
import numpy

libs = ['openblas','gomp']

arsenic_wrap=Extension('arsenic_wrap', 
        libraries = libs,
        include_dirs=[numpy.get_include()],
        language='c++',
        extra_compile_args = ['-DNDEBUG', '-DINT_64BITS', '-DAXPBY', '-fPIC', '-fopenmp', '-std=c++11', '-Wno-unused-function', '-Wno-write-strings' , '-fmax-errors=5'],
        sources=['arsenic_wrap_module.cpp'])

setup(name='arsenic-openblas',
        version='0.1',
        author_email="julien.mairal@inria.fr",
        license='bsd-3-clause',
        url="http://julien.mairal.org/arsenic/",
        description='optimization toolbox for machine learning',
#        install_requires=['openblas'],
        ext_modules=[arsenic_wrap],
        py_modules=['arsenic'])


