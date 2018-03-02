from setuptools import setup, find_packages
setup(
    name='bonndit',
    version='0.1',
    packages=find_packages(),
    scripts=['scripts/shore/shore-response.py']
    
    # Maybe also includ docutils as documentation will use reStructuredText
    install_requires=['nibabel', 'numpy', 'dipy']
    
    # Metadata for upload to PyPI
    author='',
    author_email='',
    description='The bonndit package contains the latest diffusion imaging tools developed at the University of Bonn.',
    long_description='',
    license='',
    keywords='',
    url='',   # Link to readthedocs if documenation is available or to the labs homepage

    
    )
