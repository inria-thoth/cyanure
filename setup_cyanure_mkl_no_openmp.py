from setuptools import setup, Extension
import numpy


libs = ['mkl_rt']


cyanure_wrap = Extension(
        'cyanure_wrap',
        libraries=libs,
        include_dirs=[numpy.get_include()],
        language='c++',
        extra_compile_args=[
                '-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '-fPIC',
                '-std=c++11'],
        sources=['cyanure_wrap_module.cpp'])

setup(name='cyanure-mkl-no-openmp',
      version='0.22',
      author="Julien Mairal",
      author_email="julien.mairal@inria.fr",
      license='bsd-3-clause',
      url="http://julien.mairal.org/cyanure/",
      description='optimization toolbox for machine learning',
      install_requires=['scipy'],
      ext_modules=[cyanure_wrap],
      py_modules=['cyanure'])
