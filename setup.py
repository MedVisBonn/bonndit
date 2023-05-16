#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
import numpy
import os

directive_def = get_directive_defaults()
directive_def['linetrace'] = True
directive_def['binding'] = True
if "MKLROOT" not in os.environ:
    raise Exception("""MKLROOT has to be a enviroment Variable. Follow the 
					https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup.html description to set them correctly""")

mklroot = os.environ["MKLROOT"]

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

suite_sparse_libs = ['lapack', 'ccolamd', 'spqr', 'cholmod', 'colamd', 'camd', 'amd', 'suitesparseconfig']
ceres_libs = ['glog', 'gflags']
watson_libraries = ceres_libs + suite_sparse_libs + ['pthread', 'fftw3', 'm', 'watsonfit']

ext_modules = [
    Extension("bonndit.utilc.watsonfitwrapper",
              sources=["src/bonndit/utilc/watsonfitwrapper.pyx"],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ('CYTHON_TRACE', '1')],
              include_dirs=[".", numpy.get_include(), "/usr/lib"],
              libraries=watson_libraries,
              language="c++",
              extra_compile_args=["-I.", "-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=["-L/usr/local/include", "-fopenmp", "-Wl,--no-as-needed"]
              ),
    Extension(
        "bonndit.utilc.blas_lapack",
        ["src/bonndit/utilc/blas_lapack.pyx"],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"],
        define_macros = [('CYTHON_TRACE', '1')],
),
    Extension(
        "bonndit.utilc.quaternions",
        ["src/bonndit/utilc/quaternions.pyx"],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"],
        define_macros = [('CYTHON_TRACE', '1')],
),
    Extension(
        "bonndit.utilc.cython_helpers",
        ["src/bonndit/utilc/cython_helpers.pyx"],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        define_macros=[('CYTHON_TRACE', '1')],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.utilc.hota",
        ["src/bonndit/utilc/hota.pyx"],
        extra_compile_args=["-Ofast"],
        define_macros=[('CYTHON_TRACE', '1')],
    ),
    Extension(
        "bonndit.utilc.trilinear",
        ["src/bonndit/utilc/trilinear.pyx"],
        extra_compile_args=["-Ofast"],
        define_macros=[('CYTHON_TRACE', '1')],
    ),
    Extension(
        "bonndit.utilc.structures",
        ["src/bonndit/utilc/structures.pyx"],
        extra_compile_args=["-Ofast"],
        define_macros=[('CYTHON_TRACE', '1')],
    ),
    Extension(
        "bonndit.utilc.penalty_spherical",
        ["src/bonndit/utilc/penalty_spherical.pyx"],
        extra_compile_args=["-Ofast"],
        define_macros=[('CYTHON_TRACE', '1')],
    ),
    Extension(
        "bonndit.utilc.lowrank",
        ["src/bonndit/utilc/lowrank.pyx"],
        extra_compile_args=["-Ofast"],
        define_macros=[('CYTHON_TRACE', '1')],
    ),
    Extension(
        "bonndit.directions.fodfapprox",
        ["src/bonndit/directions/fodfapprox.pyx"],
        include_dirs=[numpy.get_include(), '.'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=['-fopenmp', '-Ofast'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "bonndit.tracking.ItoW",
        ["src/bonndit/tracking/ItoW.pyx"],
        extra_compile_args=["-Ofast"],
        define_macros=[('CYTHON_TRACE', '1')],
    ),
    Extension(
        "bonndit.tracking.alignedDirection",
        ["src/bonndit/tracking/alignedDirection.pyx"],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"],
        define_macros = [('CYTHON_TRACE', '1')],
), Extension(
        "bonndit.tracking.kalman.model",
        ["src/bonndit/tracking/kalman/model.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot,"/usr/include/" ],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl", "watsonfit"],
        library_dirs=["%s/lib/intel64" % mklroot, "/usr/bin"],
        extra_compile_args=["-I.", "-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-L/usr/local/include", "-fopenmp", "-Wl,--no-as-needed"]

    ),
    Extension(
        "bonndit.tracking.kalman.kalman",
        ["src/bonndit/tracking/kalman/kalman.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", '-Ofast'],
        extra_link_args=["-Wl,--no-as-needed"]

    ),
    Extension(
        "bonndit.tracking.interpolation",
        ["src/bonndit/tracking/interpolation.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot, "/usr/lib"],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.tracking.integration",
        ["src/bonndit/tracking/integration.pyx"],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.tracking.stopping",
        ["src/bonndit/tracking/stopping.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        include_dirs=[numpy.get_include(), "%s/include" % mklroot],
        libraries=["mkl_rt", "mkl_sequential", "mkl_core", "pthread", "m", "dl"],
        library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.tracking.tracking_prob",
        ["src/bonndit/tracking/tracking_prob.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        extra_compile_args=['-fopenmp', "-Ofast"],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "bonndit.pmodels.means",
        ["src/bonndit/pmodels/means.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        extra_compile_args=["-Ofast"],
    ),
    Extension(
        "bonndit.pmodels.model_avg",
        ["src/bonndit/pmodels/model_avg.pyx"],
        extra_compile_args=["-Ofast"],
    ),
    Extension(
        "bonndit.filter.filter",
        ["src/bonndit/filter/filter.pyx"],
        extra_compile_args=["-Ofast"],
    ),
    Extension(
        "bonndit.filter.filter",
        ["src/bonndit/filter/filter.pyx"],
        extra_compile_args=["-Ofast"],
    ),
    Extension(
        "bonndit.directions.csd_peaks",
        ["src/bonndit/directions/csd_peaks.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-Ofast"],
    ),
]

requirements = ['nibabel', 'numpy', 'pandas', 'dipy', 'scipy', 'tqdm',
                'cvxopt', 'mpmath', 'plyfile', 'Cython', 'pynrrd', 'pyshtools']

setup_requirements = ['pytest-runner', 'cython']

test_requirements = ['pytest', 'nibabel', 'numpy', 'dipy', 'scipy', 'tqdm',
                     'cvxopt', 'mpmath', 'pynrrd']
print(find_packages('src'))
setup(
    author="Johannes Gruen",
    author_email='jgruen@uni-bonn.de',
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="The bonndit package contains the latest diffusion imaging tools developed at the University of Bonn.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bonndit',
    name='bonndit',

    packages=find_packages('src', exclude=('tests',)),
    package_dir={'': 'src'},
    scripts=['scripts/mtdeconv',
             'scripts/stdeconv',
             'scripts/watson-fitting',
             'scripts/kurtosis',
             'scripts/dtiselectvols',
             'scripts/low-rank-k-approx',
             'scripts/peak-modelling',
             'scripts/prob-tracking',
             'scripts/bundle-filtering',
             'scripts/csd-peaks',
             'scripts/data2fodf'],
    ext_modules=cythonize(ext_modules, compiler_directives={'boundscheck': False, 'wraparound': False,
                                                            'optimize.unpack_method_calls': False}),
    # cmdclass={'build_ext': build_ext},
    package_data={"": ['*.pxd', '*.npz', '*.npy']},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MedVisBonn/bonndit',
    version='0.2.0',
    gdb_debug=True,
    zip_safe=False,
)
