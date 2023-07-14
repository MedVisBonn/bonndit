#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

import sysconfig
import sys
import os
def path_to_build_folder():
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-cpython-{version[0]}{version[1]}/bonndit/utilc"
    dir_name = f.format(dirname='lib',
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)
    return os.path.join('build', dir_name)

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()



suite_sparse_libs = ['lapack', 'ccolamd', 'spqr', 'cholmod', 'colamd', 'camd', 'amd', 'suitesparseconfig']
ceres_libs = ['glog', 'gflags']
watson_libraries = ['m', 'watsonfit']

ext_modules = [
  #  Extension("watsonfit", sources=['src/bonndit/utilc/watsonfit.cpp'],
  #      libraries = ['cerf']),
                Extension("bonndit.utilc.watsonfitwrapper",
              sources=["src/bonndit/utilc/watsonfitwrapper.pyx",'src/bonndit/utilc/watsonfit.cpp' ],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ('CYTHON_TRACE', '1')],
              include_dirs=[".", numpy.get_include(), "/usr/lib", path_to_build_folder()],
              libraries=['cerf'],
              language="c++",
              extra_compile_args=["-I.", "-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=["-L/usr/local/include", "-fopenmp", "-Wl,--no-as-needed", "-I" + path_to_build_folder()],
              ),
    Extension(
        "bonndit.utilc.blas_lapack",
        ["src/bonndit/utilc/blas_lapack.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["cblas"],
       # library_dirs=["%s/lib/intel64" % mklroot],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
       # extra_link_args=["-Wl,--no-as-needed"],
       # define_macros = [('CYTHON_TRACE', '1')],
),
    Extension(
        "bonndit.utilc.quaternions",
        ["src/bonndit/utilc/quaternions.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["cblas"],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"],
),
    Extension(
        "bonndit.utilc.cython_helpers",
        ["src/bonndit/utilc/cython_helpers.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=['lapack'],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.utilc.hota",
        ["src/bonndit/utilc/hota.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=['cblas'],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
    ),
    Extension(
        "bonndit.utilc.trilinear",
        ["src/bonndit/utilc/trilinear.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=['cblas'],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
    ),
    Extension(
        "bonndit.utilc.structures",
        ["src/bonndit/utilc/structures.pyx"],
        extra_compile_args=["-Ofast"],
    ),
    Extension(
        "bonndit.utilc.penalty_spherical",
        ["src/bonndit/utilc/penalty_spherical.pyx"],
        extra_compile_args=["-Ofast"],
    ),
    Extension(
        "bonndit.utilc.lowrank",
        ["src/bonndit/utilc/lowrank.pyx"],
        extra_compile_args=["-Ofast"],
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
    ),
    Extension(
        "bonndit.tracking.alignedDirection",
        ["src/bonndit/tracking/alignedDirection.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=['cblas'],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"],
), Extension(
        "bonndit.tracking.kalman.model",
        ["src/bonndit/tracking/kalman/model.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),('CYTHON_TRACE', '1')],
        include_dirs=[".", numpy.get_include(),"/usr/include/" ,path_to_build_folder()],
        libraries=['cblas', "pthread", "m", "dl"],
        extra_compile_args=["-I.", "-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-L/usr/local/include", "-fopenmp", "-Wl,--no-as-needed"],
    ),
    Extension(
        "bonndit.tracking.kalman.kalman",
        ["src/bonndit/tracking/kalman/kalman.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=[numpy.get_include()],
        libraries=["lapack", "cblas", "pthread", "m", "dl"],
        extra_compile_args=["-Wall", "-m64", '-Ofast'],
        extra_link_args=["-Wl,--no-as-needed"]

    ),
    Extension(
        "bonndit.tracking.interpolation",
        ["src/bonndit/tracking/interpolation.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=[numpy.get_include(),path_to_build_folder()],
        libraries=["cblas", "pthread", "m", "dl"],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.tracking.integration",
        ["src/bonndit/tracking/integration.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["cblas", "pthread", "m", "dl"],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.tracking.stopping",
        ["src/bonndit/tracking/stopping.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=[numpy.get_include()],
        libraries=["cblas", "pthread", "m", "dl"],
        extra_compile_args=["-Wall", "-m64", "-Ofast"],
        extra_link_args=["-Wl,--no-as-needed"]
    ),
    Extension(
        "bonndit.tracking.tracking_prob",
        ["src/bonndit/tracking/tracking_prob.pyx"],
        extra_compile_args=['-fopenmp', "-Ofast"],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "bonndit.pmodels.means",
        ["src/bonndit/pmodels/means.pyx"],
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
  #  install_requires=requirements,
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
    package_data={"": ['*.pxd', '*.npz', '*.npy', '*.so', '*.h']},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MedVisBonn/bonndit',
    version='0.2.0',
    zip_safe=False,
)
