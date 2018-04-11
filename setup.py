#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['nibabel', 'numpy', 'dipy', 'scipy', 'tqdm', 'cvxopt']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'nibabel', 'numpy', 'dipy', 'scipy', 'tqdm', 'cvxopt']

setup(
    author="Olivier Morelle",
    author_email='morelle@uni-bonn.de',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
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
    scripts=['scripts/get_fodfs.py'],
    
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Oli4/bonndit',
    version='0.1.0',
    zip_safe=False,
)
