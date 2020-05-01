from setuptools import setup, Extension
import numpy
import sys, os

libs = ['mkl_rt']

pathpython=os.path.dirname(sys.executable);

cyanure_wrap = Extension(
    'cyanure_wrap',
    libraries=libs,
    include_dirs=[numpy.get_include()],
    library_dirs=[pathpython+'\\Library\\lib'],
    language='c++',
    extra_compile_args=['-DNDEBUG', '-DINT_64BITS', '-DHAVE_MKL', '-DAXPBY', '/permissive-', '/W1'],
    sources=['cyanure_wrap_module.cpp'])

setup(name='cyanure-mkl',
      version='0.21post4',
      author="Julien Mairal",
      author_email="julien.mairal@inria.fr",
      license='bsd-3-clause',
      url="http://julien.mairal.org/cyanure/",
      description='optimization toolbox for machine learning',
      install_requires=['scipy'],
      ext_modules=[cyanure_wrap],
      py_modules=['cyanure'])
