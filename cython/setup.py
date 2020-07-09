#from distutils.core import setup , Extension
#from Cython.Build import cythonize
import numpy
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

#ext_modules=[
#    Extension("three_functions",
#              ["three_functions.pyx"],
#              libraries=["m"],
#              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#              extra_link_args=['-fopenmp']
#              )
#]

#setup(
#    name="three_functions",
#    ext_modules = cythonize('three_functions.pyx'),
#    include_dirs=[numpy.get_include()]
#)
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('bmn/utils.pyx'),
      include_dirs=[numpy.get_include()])
