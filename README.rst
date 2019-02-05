=======
bonndit
=======


.. image:: https://img.shields.io/pypi/v/bonndit.svg
        :target: https://pypi.python.org/pypi/bonndit

.. image:: https://travis-ci.com/MedVisBonn/bonndit.svg?token=FqhWHkwE6EPeAZDNegqx&branch=master
    :target: https://travis-ci.com/MedVisBonn/bonndit

.. image:: https://readthedocs.org/projects/bonndit/badge/?version=latest
        :target: https://bonndit.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


The bonndit package contains the latest diffusion imaging tools developed at the University of Bonn. This early release focuses on our framework for single and multi tissue deconvolution using constrained higher-order tensor fODFs. In addition, it includes code for fitting the Diffusional Kurtosis (DKI) model. More will follow as time permits.


* Free software: GNU General Public License v3
* Documentation: https://bonndit.readthedocs.io.

Installation
------------
To install bonndit, run the following command

.. code-block:: console

    $ pip install bonndit

Features
--------
* :code:`stdeconv`: Script to calculate white matter response function and fiber orientation distribution functions (fODFs) from a single shell diffusion signal.
* :code:`mtdeconv`: Script to calculate multi tissue response functions and fiber orientation distribution functions (fODFs) from multi shell or DSI signals.
* :code:`kurtosis`: Script to fit a kurtosis model using quadratic cone programming to guarantee a minimum diffusivity. It also calculates kurtosis measures based on the fitted model.
* All functionality implemented in an object oriented manner.
* Multiprocessing support implemented in the underlying infrastructure for faster computations
* Logs written to your output directory


Getting Started
---------------
For calculating the multi-tissue response functions and the fODFs for given data run the following command:

.. code-block:: console

    $ mtdeconv -i /path/to/your/data --verbose True

The specified folder should contain the following files:

* :code:`bvecs`: b-vectors
* :code:`bvals`: b-values
* :code:`data.nii.gz`: The diffusion weighted data
* :code:`dti_FA.nii.gz`: Diffusion tensor Fractional Anisotropy map
* :code:`fast_pve_0.nii.gz`: CSF mask
* :code:`fast_pve_1.nii.gz`: GM mask
* :code:`fast_pve_2.nii.gz`: WM mask
* :code:`dti_V1.nii.gz`: The first eigenvector of the diffusion tensor

The :code:`dti_*` files can be generated using FSL's :code:`dtifit`. The :code:`fast_*` files can be generated from coregistered T1 weighted images using FSL's :code:`fast`.

Optional, but recommended to greatly speed up computation:

* :code:`mask.nii.gz`: Binary mask, specifying brain voxels in which to estimate the model


If you want to see a list of parameters type the following:

.. code-block:: console

    $ mtdeconv -h

Reference
----------

If you use our software as part of a scientific project, please cite the corresponding publications. The method implemented in :code:`stdeconv` and :code:`mtdeconv` was first introduced in

* Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Fast and Accurate Multi-Tissue Deconvolution Using SHORE and H-psd Tensors. In: Proc. Medical Image Analysis and Computer-Aided Intervention (MICCAI) Part III, pp. 502-510, vol. 9902 of LNCS, Springer, 2016

It was refined and extended in

* Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Versatile, Robust, and Efficient Tractography With Constrained Higher-Order Tensor fODFs. In: Int'l J. of Computer Assisted Radiology and Surgery, 12(8):1257-1270, 2017

The use of quadratic cone programming to make the kurtosis fit more stable which is implemented in :code:`kurtosis` has been explained in the methods section of

* Samuel Groeschel, G. E. Hagberg, T. Schultz, D. Z. Balla, U. Klose, T.-K. Hauser, T. Nägele, O. Bieri, T. Prasloski, A. MacKay, I. Krägeloh-Mann, K. Scheffler: Assessing white matter microstructure in brain regions with different myelin architecture using MRI. In: PLOS ONE 11(11):e0167274, 2016

PDFs can be obtained from the respective publisher, or the academic homepage of Thomas Schultz: http://cg.cs.uni-bonn.de/en/people/prof-dr-thomas-schultz/

Authors
-------

* **Michael Ankele** - *Initial work* - [momentarylapse] (https://github.com/momentarylapse)

* **Thomas Schultz** - *Initial work* - [ThomasSchultz] (https://github.com/ThomasSchultz)

* **Olivier Morelle** - *Code curation, documentation and test* [Oli4] (https://github.com/Oli4)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
