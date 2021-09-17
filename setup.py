#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

ext_modules = [
    Extension(
        "bonndit.helper_functions.cython_helpers",
        ["src/bonndit/helper_functions/cython_helpers.pyx"],
    ),
    Extension(
        "bonndit.helper_functions.hota",
        ["src/bonndit/helper_functions/hota.pyx"],
    ),
    Extension(
        "bonndit.helper_functions.structures",
        ["src/bonndit/helper_functions/structures.pyx"],
    ),
    Extension(
        "bonndit.helper_functions.penalty_spherical",
        ["src/bonndit/helper_functions/penalty_spherical.pyx"],
    ),
    Extension(
        "bonndit.helper_functions.average",
        ["src/bonndit/helper_functions/average.pyx"],
    ),
    Extension(
        "bonndit.directions.fodfapprox",
        ["src/bonndit/directions/fodfapprox.pyx"],
        include_dirs=[numpy.get_include(), '.'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "bonndit.tracking.ItoW",
        ["src/bonndit/tracking/ItoW.pyx"],
    ),
    Extension(
        "bonndit.tracking.alignedDirection",
        ["src/bonndit/tracking/alignedDirection.pyx"],
        include_dirs=[numpy.get_include(), '.'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "bonndit.tracking.interpolation",
        ["src/bonndit/tracking/interpolation.pyx"],
        include_dirs=[numpy.get_include(), '.'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
	Extension(
		"bonndit.tracking.integration",
		["src/bonndit/tracking/integration.pyx"],
	),
	Extension(
        "bonndit.tracking.stopping",
        ["src/bonndit/tracking/stopping.pyx"],
    ),
    Extension(
        "bonndit.tracking.tracking_prob",
        ["src/bonndit/tracking/tracking_prob.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "bonndit.pmodels.means",
        ["src/bonndit/pmodels/means.pyx"],
    ),
    Extension(
        "bonndit.pmodels.model_avg",
        ["src/bonndit/pmodels/model_avg.pyx"],
    ),
    Extension(
        "bonndit.filter.filter",
        ["src/bonndit/filter/filter.pyx"],
    ),
    Extension(
        "bonndit.filter.filter",
        ["src/bonndit/filter/filter.pyx"],
    ),
    Extension(
        "bonndit.directions.csd_peaks",
        ["src/bonndit/directions/csd_peaks.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],

    ),
]

requirements = ['nibabel', 'numpy', 'pandas', 'dipy', 'scipy', 'tqdm',
                'cvxopt', 'mpmath', 'plyfile', 'Cython']

setup_requirements = ['pytest-runner', 'cython']

test_requirements = ['pytest', 'nibabel', 'numpy', 'dipy', 'scipy', 'tqdm',
                     'cvxopt', 'mpmath']
print(find_packages('src'))
setup(
    author="Olivier Morelle",
    author_email='morelle@uni-bonn.de',
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
			 'scripts/kurtosis',
			 'scripts/dtiselectvols',
			 'scripts/low-rank-k-approx',
			 'scripts/peak-modelling',
			 'scripts/prob-tracking',
			 'scripts/bundle-filtering',
			 'scripts/csd-peaks'],
    ext_modules=cythonize(ext_modules),
    # cmdclass={'build_ext': build_ext},
    package_data={"": ['*.pxd', '*.npz']},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MedVisBonn/bonndit',
    version='0.2.0',
    zip_safe=False,
)
