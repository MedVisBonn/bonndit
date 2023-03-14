=======
bonndit
=======


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

Installation
------------
To install bonndit, run the following command

.. code-block:: console

    $ pip install bonndit

Features
--------
* `stdeconv <https://bonndit.readthedocs.io/en/latest/readme.html#stdeconv>`_: Script to calculate white matter response function and fiber orientation distribution functions (fODFs) from a single shell diffusion signal.
* `mtdeconv <https://bonndit.readthedocs.io/en/latest/readme.html#mtdeconv>`_: Script to calculate multi tissue response functions and fiber orientation distribution functions (fODFs) from multi shell or DSI signals.
* `kurtosis <https://bonndit.readthedocs.io/en/latest/readme.html#kurtosis>`_: Script to fit a kurtosis model using quadratic cone programming to guarantee a minimum diffusivity. It also calculates kurtosis measures based on the fitted model.
* `low-rank-k-approx <https://bonndit.readthedocs.io/en/latest/readme.html#low-rank-k-approx>`_: Script to calculate the low rank approx of a symmetric 4th order tensor.
* `csd-peaks <https://bonndit.readthedocs.io/en/latest/readme.html#csd-peaks>`_: Script to extract the peaks from a 8th order tensor.
* `peak-modelling <https://bonndit.readthedocs.io/en/latest/readme.html#peak-modelling>`_: Script to fuse multiple rank k=1,2,3 models into one new model with reduced model uncertainty.
* `prob-tracking <https://bonndit.readthedocs.io/en/latest/readme.html#prob-tracking>`_: Script for probabilistic tracking.
* `bundle-filtering <https://bonndit.readthedocs.io/en/latest/readme.html#bundle-filtering>`_: Script to apply various filters to the streamlines.
* Multiprocessing support implemented in the underlying infrastructure for faster computations
* Logs written to your output directory



Reference
----------

If you use our software as part of a scientific project, please cite the corresponding publications. The method implemented in :code:`stdeconv` and :code:`mtdeconv` was first introduced in

* Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Fast and Accurate Multi-Tissue Deconvolution Using SHORE and H-psd Tensors. In: Proc. Medical Image Analysis and Computer-Aided Intervention (MICCAI) Part III, pp. 502-510, vol. 9902 of LNCS, Springer, 2016

It was refined and extended in

* Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Versatile, Robust, and Efficient Tractography With Constrained Higher-Order Tensor fODFs. In: Int'l J. of Computer Assisted Radiology and Surgery, 12(8):1257-1270, 2017

The methods implemented in :code:`low-rank-k-approx` was first introduced in

* Thomas Schultz, Hans-Peter Seidel: Estimating Crossing Fibers: A Tensor Decomposition Approach. In: IEEE Transactions on Visualization and Computer Graphics, 14(6):1635-42, 2008

The methods implemented in :code:`peak-modelling` was first introduced in

* Johannes Grün, Gemma van der Voort, Thomas Schultz: Reducing Model Uncertainty in Crossing Fiber Tractography. In proceedings of EG Workshop on Visual Computing for Biology and Medicine, pages 55-64, 2021

Extended in:

* Johannes Grün, Gemma van der Voort, Thomas Schultz: Model Averaging and Bootstrap Consensus Based Uncertainty Reduction in Diffusion MRI Tractography. In: Computer Graphics Forum, 2022

The regularized tractography methods (joint low-rank and low-rank UKF) were first implemented in :code:`prob-tracking` and introduced in

* Johannes Grün, Samuel Gröschel, Thomas Schultz: Spatially Regularized Low-Rank Tensor Approximation for Accurate and Fast Tractography. In NeuroImage, 2023


The use of quadratic cone programming to make the kurtosis fit more stable which is implemented in :code:`kurtosis` has been explained in the methods section of

* Samuel Groeschel, G. E. Hagberg, T. Schultz, D. Z. Balla, U. Klose, T.-K. Hauser, T. Nägele, O. Bieri, T. Prasloski, A. MacKay, I. Krägeloh-Mann, K. Scheffler: Assessing white matter microstructure in brain regions with different myelin architecture using MRI. In: PLOS ONE 11(11):e0167274, 2016

PDFs can be obtained from the respective publisher, or the academic homepage of Thomas Schultz: http://cg.cs.uni-bonn.de/en/people/prof-dr-thomas-schultz/

Authors
-------

* **Michael Ankele** - *Initial work* - [momentarylapse] (https://github.com/momentarylapse)

* **Thomas Schultz** - *Initial work* - [ThomasSchultz] (https://github.com/ThomasSchultz)

* **Olivier Morelle** - *Code curation, documentation and test* [Oli4] (https://github.com/Oli4)

* **Johannes Grün** - *Extended work* - [JoGruen] (https://github.com/JoGruen)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
