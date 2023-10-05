Welcome to bonndit's documentation!
===================================

The bonndit package contains computational tools for diffusion MRI processing developed at the University of Bonn. It is free software, published under the GNU General Public License v3.

bonndit implements constrained single and multi tissue deconvolution with higher-order tensor fODFs [Ankele17]_, and the extraction of principal fiber directions with low-rank tensor approximation [Schultz08]_. It also includes code for fiber tractography based on higher-order tensor fODFs, and for filtering the resulting set of streamlines. In particular, bonndit implements spatially regularized tracking using joint tensor decomposition or an Unscented Kalman Filter [Gruen23]_. It also contains code from a study in which we compared the strategy of selecting the most suitable number of fiber compartments per voxel to an adaptive model averaging which reduced the model uncertainty [Gruen22]_.

Finally, the package includes code for suitably constrained fitting of the Diffusional Kurtosis (DKI) model, and computation of corresponding invariants [Groeschel16]_.

These features are provided by the following scripts, which are documented in more detail below. A tutorial for performing fiber tracking with our code is available in :ref:`Getting Started`

* :ref:`stdeconv`: Calculates white matter response function and fiber orientation distribution functions (fODFs) from a single shell diffusion imaging.
* :ref:`mtdeconv`: Calculates multi tissue response functions and fiber orientation distribution functions (fODFs) from multi shell imaging or DSI.
* :ref:`low-rank-k-approx`: Calculates the low rank approx of a symmetric 4th order tensor.
* :ref:`csd-peaks`: Extracts the peaks from an 8th order tensor.
* :ref:`peak-modelling`: Fuses multiple rank k=1,2,3 models into one new model with reduced model uncertainty.
* :ref:`prob-tracking`: Performs probabilistic tractography.
* :ref:`bundle-filtering`: Applies various filters to the streamlines.
* :ref:`kurtosis`: Fits a kurtosis model using quadratic cone programming to guarantee a minimum diffusivity. It also calculates kurtosis measures based on the fitted model.
  
All scripts help keep track of computations and parameters by writing log files along with their primary output. Many also support multiprocessing for faster computations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   gettingstarted
   scripts
   contributing
   authors
   references
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
