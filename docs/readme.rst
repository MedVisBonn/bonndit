========
bonndit
========


.. image:: https://img.shields.io/pypi/v/bonndit.svg
        :target: https://pypi.python.org/pypi/bonndit

.. image:: https://readthedocs.org/projects/bonndit/badge/?version=latest
        :target: https://bonndit.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


The bonndit package contains the latest diffusion imaging tools developed at the University of Bonn.
Within the package the whole pipeline from the raw data to tractogaphy is covered.

Included is a framework for
single and multi tissue deconvolution using constrained higher-order tensor fODFs.
In addition, it includes code for fitting the Diffusional Kurtosis (DKI) model. To extract directions two methods are
included. On the one hand simple peak extraction and on the other hand a low rank approximation framework.
Further an average and selection approach is implemented, which reduces the model uncertainty. Within
these multivectorfields we can apply a probabilistic streamline based tracking approach, which can be filtered afterwards.


* Free software: GNU General Public License v3
* Documentation: https://bonndit.readthedocs.io.


Features
--------
* :code:`stdeconv`: Script to calculate white matter response function and fiber orientation distribution functions (fODFs) from a single shell diffusion signal.
* :code:`mtdeconv`: Script to calculate multi tissue response functions and fiber orientation distribution functions (fODFs) from multi shell or DSI signals.
* :code:`kurtosis`: Script to fit a kurtosis model using quadratic cone programming to guarantee a minimum diffusivity. It also calculates kurtosis measures based on the fitted model.
* :code:`low-rank-k-approx`: Script to calculate the low rank approx of a symmetric 4th order tensor.
* :code:`csd-peaks`: Script to extract the peaks from a 8th order tensor.
* :code:`peak-modelling`: Script to fuse multiple rank k=1,2,3 models into one new model with reduced model uncertainty.
* :code:`prob-tracking`: Script for probabilistic tracking.
* :code:`bundle-filtering`: Script to apply various filters to the streamlines.
* Multiprocessing support implemented in the underlying infrastructure for faster computations
* Logs written to your output directory





