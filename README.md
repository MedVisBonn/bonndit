# bonndit: diffusion imaging tools developed in Bonn


**This package is still under development!**

## Current state
+ You can compute the shore response functions for gray matter, white matter and
csf using bonndits ShoreModel and ShoreFit classes. You have the same functionality
using the `get_fodfs.py` script which is installed with bonndit.

## Next steps
+ Deconvolution with response functions and saving of the results
+ Better Documentation
+ Tests
+ Kurtosis module

## Installation

This module isn't available from PyPI yet. To use and test the current state
execute the following in the directory containing the cloned repository. This
installs bonndit in development mode. The current version has not been tested
with Python 2.

``` bash
pip install -e bonndit
```

## Getting Started

This [notebook](notebooks/shore_example.ipynb) in the notebooks folder
contains an example of how to use
bonndits shore module. Another possibility is to use the `get_fodfs.py` script
which is installed with bonndit. To run the script with the default parameters you
only need to specify the directory which contains all needed input files. If you set
verbose to True you will see a progress bar for the computation of each response
function.

``` bash
get_fodfs.py -i /path/to/your/folder --verbose=True
```
