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


The bonndit package contains the latest diffusion imaging tools developed at the University of Bonn.


* Free software: GNU General Public License v3
* Documentation: https://bonndit.readthedocs.io.

Installation
------------
To install bonndit, run the following commands. As long as the package is not public you need to provide your github credentials for cloning.

.. code-block:: console

    $ git clone git@github.com:MedVisBonn/bonndit.git
    $ pip install -e bonndit

In future an installation with :code:`pip` directly from PyPI will be also available.

Features
--------

* :code:`get_fodfs.py`: Script to calculate response functions and fiber orientation distribution functions(fODFs) from the diffusion signal.

Getting Started
---------------
For calculating the response functions and the fODFs from the given data run the following command:

.. code-block:: console

    $ get_fodfs.py -i /path/to/your/data --verbose True

The specified folder should contain the following files:

* `bvecs`: b-vectors
* `bvals`: b-values
* `data.nii.gz`: The diffusion weighted data
* `dti_FA.nii.gz`: Diffusion tensor fractional anisotropy map
* `fast_pve_0.nii.gz`: CSF mask
* `fast_pve_1.nii.gz`: GM mask
* `fast_pve_2.nii.gz`: WM mask
* `dti_V1.nii.gz`: The first eigenvector of the diffusion tensor

Optional:

* `mask.nii.gz`: DTI mask


If you want to see a list of parameters type the following:

.. code-block:: console

    $ get_fodfs.py -h


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
