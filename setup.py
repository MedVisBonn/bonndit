#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['nibabel', 'numpy', 'pandas', 'dipy', 'scipy', 'tqdm',
                'cvxopt', 'mpmath']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'nibabel', 'numpy', 'dipy', 'scipy', 'tqdm',
                     'cvxopt', 'mpmath']

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
    scripts=['scripts/mtdeconv', 'scripts/stdeconv', 'scripts/kurtosis'],

    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MedVisBonn/bonndit',
    version='0.1.2',
    zip_safe=False,
)
