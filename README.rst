=======
bonndit
=======


.. image:: https://badge.fury.io/py/bonndit.svg
    :target: https://badge.fury.io/py/bonndit

.. image:: https://readthedocs.org/projects/bonndit/badge/?version=latest
        :target: https://bonndit.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

The bonndit package contains computational tools for diffusion MRI processing developed at the University of Bonn.

bonndit implements constrained single and multi tissue deconvolution with higher-order tensor fODFs [Ankele17]_, and the extraction of principal fiber directions with low-rank tensor approximation [Schultz08]_. It also includes code for fiber tractography based on higher-order tensor fODFs, and for filtering the resulting set of streamlines. In particular, bonndit implements spatially regularized tracking using joint tensor decomposition or an Unscented Kalman Filter [Gruen23]_. It also contains code from a study in which we compared the strategy of selecting the most suitable number of fiber compartments per voxel to an adaptive model averaging which reduced the model uncertainty [Gruen22]_.

Finally, the package includes code for suitably constrained fitting of the Diffusional Kurtosis (DKI) model, and computation of corresponding invariants [Groeschel16]_.


* Free software: GNU General Public License v3
* Documentation: https://bonndit.readthedocs.io.

Installation
------------
To install bonndit via pip, run the following command

.. code-block:: console

    $ pip install bonndit

To install bonndit via conda, run

.. code-block:: console

    $ conda install bonndit -c xderes -c conda-forge
    
Features
--------
An overview of the scripts and functionality included in bonndit is given in `our documentation <https://bonndit.readthedocs.io/en/latest/>`_. It also includes `a tutorial for performing fiber tracking with our code <https://bonndit.readthedocs.io/en/latest/gettingstarted.html>`_.

Reference
----------

If you use our software as part of a scientific project, please cite the corresponding publications. The method implemented in :code:`stdeconv` and :code:`mtdeconv` was first introduced in

.. [Ankele16] Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Fast and Accurate Multi-Tissue Deconvolution Using SHORE and H-psd Tensors. In: Proc. Medical Image Analysis and Computer-Aided Intervention (MICCAI) Part III, pp. 502-510, vol. 9902 of LNCS, Springer, 2016

It was refined and extended in

.. [Ankele17] Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Versatile, Robust, and Efficient Tractography With Constrained Higher-Order Tensor fODFs. In: Int'l J. of Computer Assisted Radiology and Surgery, 12(8):1257-1270, 2017

The methods implemented in :code:`low-rank-k-approx` was first introduced in

.. [Schultz08] Thomas Schultz, Hans-Peter Seidel: Estimating Crossing Fibers: A Tensor Decomposition Approach. In: IEEE Transactions on Visualization and Computer Graphics, 14(6):1635-42, 2008

The methods implemented in :code:`peak-modelling` was first introduced in

.. [Gruen21] Johannes Grün, Gemma van der Voort, Thomas Schultz: Reducing Model Uncertainty in Crossing Fiber Tractography. In proceedings of EG Workshop on Visual Computing for Biology and Medicine, pages 55-64, 2021

Extended in:

.. [Gruen22] Johannes Grün, Gemma van der Voort, Thomas Schultz: Model Averaging and Bootstrap Consensus Based Uncertainty Reduction in Diffusion MRI Tractography. In: Computer Graphics Forum 42(1):217-230, 2023

The regularized tractography methods (joint low-rank and low-rank UKF) were first implemented in :code:`prob-tracking` and introduced in

.. [Gruen23] Johannes Grün, Samuel Gröschel, Thomas Schultz: Spatially Regularized Low-Rank Tensor Approximation for Accurate and Fast Tractography. In NeuroImage 271:120004, 2023


The use of quadratic cone programming to make the kurtosis fit more stable which is implemented in :code:`kurtosis` has been explained in the methods section of

.. [Groeschel16] Samuel Groeschel, G. E. Hagberg, T. Schultz, D. Z. Balla, U. Klose, T.-K. Hauser, T. Nägele, O. Bieri, T. Prasloski, A. MacKay, I. Krägeloh-Mann, K. Scheffler: Assessing white matter microstructure in brain regions with different myelin architecture using MRI. In: PLOS ONE 11(11):e0167274, 2016

PDFs can be obtained from the respective publisher, or the academic homepage of Thomas Schultz: https://cg.cs.uni-bonn.de/person/prof-dr-thomas-schultz

Authors
-------

* **Michael Ankele** - *Constrained spherical deconvolution with tensor fODFs* - [momentarylapse] (https://github.com/momentarylapse)

* **Johannes Grün** - *Fiber tracking with spatial regularization or model averaging* - [JoGruen] (https://github.com/JoGruen)

* **Olivier Morelle** - *Code curation, documentation and testing* [Oli4] (https://github.com/Oli4)

* **Thomas Schultz** - *DKI fitting, supervision and contributions throughout* - [ThomasSchultz] (https://github.com/ThomasSchultz)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
