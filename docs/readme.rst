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
* :code:`low-rank-k-approx`: Script to calculate the low rank approx of a symmetric 4th order tensor.
* :code:`csd-peaks`: Script to extract the peaks from a 8th order tensor.
* :code:`peak-modelling`: Script to fuse multiple rank k=1,2,3 models into one new model with reduced model uncertainty.
* :code:`prob-tracking`: Script for probabilistic tracking.
* :code:`bundle-filtering`: Script to apply various filters to the streamlines.
* Multiprocessing support implemented in the underlying infrastructure for faster computations
* Logs written to your output directory


Getting Started
------------------

This introduction will help a new user to reconstruct the right CST tract of a HCP subject. For a more detailed description of all commands we refer to the next sections.

**To run the following tutorial it is necessary to have FSL installed**

First, you need to download HCP diffusion MRI data from the HCP website. This dataset includes high-quality diffusion MRI images and preprocessed data. You can download the data for free after registering on the HCP db website (db.humanconnectome.org/) and agreeing to the licence.
To follow the tutorial please download patient 904044. Therfore, search for 904044, click download image and select the `Diffusion Preporocessed`.

As an intermediate step we reconstruct fODFs as it was proposed by Ankele et al. [2]_. In this method the single fiber response function (which is then used for deconvolution) is estimated using voxels with an high fractional anisotropy.
To estimate the FA values we are selecting first high b-values (>1500) and the corresponding b vectors and measurements via:

.. code-block:: console

    dtiselectvols --outdata "HCPdir/dtidata.nii.gz" --outbvecs "HCPdir/dtibvecs" --outbvals "HCPdir/dtibvals" --indata "HCPdir/data.nii.gz" --inbvecs "HCPdir/bvecs" --inbvals "HCPdir/bvals"

Next we are estimating tensors via FSLs'

.. code-block:: console

    dtifit -k "HCPdir/dtidata.nii.gz" -o "HCPdir/dti" -m "HCPdir/mask.nii.gz" -r "HCPdir/dtibvecs" -b "HCPdir/dtibvals" -w

The `dti_FA.nii.gz` contains the FA values and can be inspected with fsleyes.

As a second step we are segmenting the T1 image into WM/GM/CSF. Therefore, we apply the mask to it via

.. code-block:: console

    fslmaths "HCPdir/T1w_acpc_dc_restore_1.25.nii.gz" -mas "HCPdir/mask.nii.gz" "HCPdir/T1_masked.nii.gz"

and running the segmentation via

.. code-block:: console

    fast -o "HCPdir/fast" "HCPdir/T1_masked.nii.gz"

Computed all prerequisites the fODFs can be estimated using

.. code-block:: console

    mtdeconv -o "HCPdir/mtdeconv/" -v True -k rank1 -r 4 -C hpsd "HCPdir"

The `fodf.nrrd` contians the computed fODFs. It can be transformed into dipy/mrtrix format using the

.. code-block:: console

    bonndit2mrtrix fodf.nrrd

command, which will output a `fodf.nii.gz` file readable by MRtrixs' mrview etc. From the fODFs we can extract fiber orientations
via [3]_

.. code-block:: console

    low-rank-k-approx "HCPdir/mdtdeconv/fodf.nrrd" "HCPdir/mtdeconv/rank3.nrrd" -r 3

As a final step we reconstruct a fiber bundle of the right CST. Therefore, we supplied pregenerated seed points with initial directions \
in the `bonndit/tests/cst-right.pts` file. For more information about the file format have a look into the tracking section.

To run the easiest version of the tractography code we run the following command:

.. code-block:: console

    prob-tracking -i "HCPdir/mtdeconv/" --seedpoints "bonndit/tests/cst-right.pts" -o "cst_unconstrained.tck"

It uses an iterative tractography approach beginning at each seedpoint into both directions. If no direction is specified in the seed file it will \
use the main direction of low-rank approximation at the closest voxel. Now it will track iteratively into both directions. Each iteration steps \
contains the following parts. First the fODF at the current point is interpolated trilinearly from its surrounding. From the fODF we are \
calculating the low-rank approximation [3]_ and choosing the next direction probabilistically. Using a Runge-Kutta integration scheme \
we are doing a step with half step size and redo the trilinear interpolation and direction choice to use the mean direction with full step size.
This is done until a stopping criteria is reached, which are set to a minimum wm density of 0.3 and a maximum curvature of 130 degrees over the last 30mm.

To run the more advanced joint low-rank approximation we have to specify

.. code-block:: console

    prob-tracking -i "HCPdir/mtdeconv/" --seedpoints "bonndit/tests/cst-right.pts" -o "cst_constrained.tck"

Instead of using the low-rank approximation, we are using a regularised version of it the joint low-rank approximation, which was introduced in [5]_ \
as first method.

To run the low-rank UKF we have to add the "ukf" flag.

.. code-block:: console

    prob-tracking -i "HCPdir/mtdeconv/" --seedpoints "bonndit/tests/cst-right.pts" -o "cst_ukf.tck" --ukf "lowrank"

We have replaced the low-rank approximation with an UKF approach which estimated the new low-rank approximation depending on the past and regularize \
through this. This was introduced in [5]_ as second approach.

More details about various options can be found below.

mtdeconv
~~~~~~~~~~~~

For calculating the multi-tissue response functions and the fODFs for given data run the following command:

.. code-block:: console

    $ mtdeconv -i /path/to/your/data

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

* :code:`mask.nii.gz`: Binary mask, specifying brain voxels in which to estimate the model.

Further important parameters are:

* :code:`--rank`: The rank of the computed fODF. Higher ranks lead to sharper peaks as well as higher susceptibility to noise. Default: 4. Supported: 4,6,8
* :code:`--kernel`: The single tissue response kernel used to estimate the CSD. Options are either `rank1` a single fiber rank-1 fiber corresponding to the rotational harmonic parts of the spherical harmonics up to the specified order or `delta` a single peak. `rank1` reduces the susceptibility to noise. Default: rank1


If you want to see a list of parameters type the following:

.. code-block:: console

    $ mtdeconv -h


low-rank-k-approx
~~~~~~~~~~~~~~~~~~~~
For calculating the low-rank k approximation of a given 4th order fODF run the following command:

.. code-block:: console

    $ low-rank-k-approx /path/to/your/fODF.nrrd path/to/output.nrrd

With additional argument

* :code:`-r`: Rank of low rank approximation. Default = 3

The output is a dim (4,r,x,y,z) array. Here the first axis contains in place 0: :math:`\lambda` the volume fraction
and in the remaining places :math:`\mathbb{v}` the unit direction.


peak-modelling
~~~~~~~~~~~~~~
Build a new model (Selection or Averaging) from given low rank k = 1,2,3 approximations by running the following command:

.. code-block:: console

    $ peak-modelling -f path/to/fODF.nrrd -i path/to/rank-1.nrrd path/to/rank-2.nrrd path/to/rank-3.nrrd -o path/to/outfile.nrrd

Further the parameters can be set

* :code:`-t`: selection or averaging. Default averaging
* :code:`-a`: a parameter for Kumaraswarmy PDF. Default a = 1
* :code:`-b`: b parameter for Kumaraswarmy PDF. Default b = 20

The output is a dim (4,3,x,y,z) array. Here the first axis contains in place 0: :math:`\lambda` the volume fraction
and in the remaining places :math:`\mathbb{v}` the unit direction. If a voxel contains only 1 or 2 directions they are
at the first entries of the second axis.

csd-peaks
~~~~~~~~~~~~~~~~
For extracting maxima from a given 8th order fODF (These can be generated by running the mtdeconv script with additional parameters -k delta -r 8 -C nonneg) by run the following command:

.. code-block:: console

    $ csd-peaks path/to/fODF.nrrd path/to/output.nrrd

Further parameters can be set

* :code:`-r`: Count of fibers to extract. Default 3
* :code:`-sa`: Minimum separation angle in degrees. Default 0
* :code:`-m`: Minimum height of peak. Default 0

The output is a dim (4,r,x,y,z) array. Here the first axis contains in place 0: :math:`\lambda` the volume fraction
and in the remaining places :math:`\mathbb{v}` the unit direction.

prob-tracking
~~~~~~~~~~~~~~~~~~
Within the generated multivectorfields, which are an output of either low-rank-approx, peak-modelling or csd-peaks,
running the following command

.. code-block:: console

    $ prob-tracking --i /path/to/inputs -o /path/to/output.ply

generates streamlines for each seed point.
If not further specified the input folder has to contain the following:

- rank3.nrrd
    Multidirectionfield, where the first dimension defines the length and the
    unit direction of the vector, second dimension defines different directions
    and remaining dimensions diffine the coordinate.

    If the file is named differently, use the `--infile` argument

- wmvolume.nrrd
    The white matter mask, which is an output of mtdeconv.

    If the file is named differently, use the `--wmmask` argument

- seedpoint.pts
    The seed point file in world coordinates. First 3 dimensions of row give
    world coordinates. Additionally a initial direction can be set by appending
    3 columns to each row denoting the direction in (x,y,z) space.

    If the file is named differently, use the `--seedpoint` argument.

If the -ukf flag is set, the input folder should also contain:

- bvals
    A text file which contains the bvals for each gradient direction.

    If the file is namend differenty, use the `--ukf_bvals` argument

- bvecs
    A text file which contains all gradient directions in the format Ax3
    If the file is named differently, use the `--ukf_bvecs` argument
- data.nrrd
    The file with the data. If the `--ukfmethod` flag is set to

    - MultiTensor it should be the raw data.

    - LowRank it should be the fodf.nrrd output from mtdeconv

    If the file is named differently, use the --ukf_data argument.

- baseline.nrrd
    File with b0 measurements

    If the file is named differently, use the `--ukf_baseline` argument

If the -disk flag is set and we want to append to a file, the inputfolder should contain
    - output.txt
        A textfile with the streamlines generated so far.
        If the file is named differently, use the `--disk_append` argument.


The output file is in ply format, which contain the vertex coordinates and the length of each streamline.

Further parameters can be set:

* :code:`--infile`: 5D (4,3,x,y,z) Multivectorfield, where the first dimension gives the length and the direction of the vector, the second dimension denotes different directions.
* :code:`--wmvolume`: WM Mask - output of mtdeconv
* :code:`--act`: 5tt output of 5ttgen. Will perform act if supplied.
* :code:`--seedpoints`: Seedspointfile: Each row denotes a seed point, where the first  3 columns give the seed point in (x,y,z). Further 3 additional columns can specified to define a initial direction. Columns should be seperated by whitespace.
* :code:`--wmmin`: Minimum WM density before tracking stops, default=0.15
* :code:`--sw_save`: Only each x step is saved. Reduces memory consumption greatly, default=1
* :code:`--sw`: Stepwidth for Euler integration, default=0.9
* :code:`--o`: Filename for output file in ply or tck format. Only ply is fully supported.
* :code:`--mtlength`: Maximum track steps, default=300
* :code:`--samples`: Samples per seed, default=1
* :code:`--max_angle`: Max angle over the last 30 mm of the streamline, default=130
* :code:`--var`: Variance for probabilistic direction selection, default=1
* :code:`--exp`: Expectation for probabilistic direction selection, default=0
* :code:`--interpolation`: decide between FACT interpolation and Trilinear interpolation.', default='Trilinear'
* :code:`--sigma_1`: Only useful if interpolation is set to TrilinearFODF and dist>0. Controls sigma1 for low-rank, default=1
* :code:`--data`: Only useful if interpolation is set to TrilinearFODF and dist>0. Controls sigma1 for low-rank
* :code:`--sigma_2`: Only useful if interpolation is set to TrilinearFODF and dist>0. Controls sigma2 for low-rank, default=1
* :code:`--dist`: Only useful if interpolation is set to TrilinearFODF. Radius of points to include, default=0
* :code:`--rank`: Only useful if interpolation is set to TrilinearFODF. Rank of low-rank approx.', default=3
* :code:`--integration`: Decide between Euler integration and Euler integration, default='Euler'
* :code:`--prob`: Decide between Laplacian, Gaussian, Scalar, ScalarNew, Deterministic and Deterministic2, default='Gaussian')
* :code:`--disk`: Write streamlines to file instead of using ram, default=True
* :code:`--disk_file`: Name of disk file. If not set a random filename is chosen.
* :code:`--disk_delete`: Delete file after finish. Otherwise further Streamlines can be appended if more streamlines are needed.
* :code:`--ukf`: The following arguments are just important if the --ukf flag is set to MultiTensor or LowRank
* :code:`--ukf_data`: File containing the raw data for ukf.
* :code:`--ukf_bvals`: File containg the bvals for each gradient direction
* :code:`--ukf_bvecs`: File containg the bvecs
* :code:`--ukf_baseline`: File containg the baseline'
* :code:`--ukf_fodf_order`: order of fODF. Only 4 and 8 are supported, default=4
* :code:`--ukf_dim_model`: Dimensions of model
* :code:`--ukf_pnoise`: Process noise
* :code:`--ukf_mnoise`: Measurement noise

For a set of possible directions :math:`v_i` and a given current direction :math:`w`, the probability of the next direction is given either by
Gaussian

.. math::

    p \left( v_i \right) = \exp \left(  -1/2 \left(\frac{  \theta_i - b }{\sigma } \right)^2 \right)

Laplacian

.. math::

    p \left( v_i \right) = 1/2 \exp \left( - \left| \frac{  \theta_i - b } { \sigma } \right| \right)

ScalarOld

.. math::

    p \left( v_i \right) =  \mathbb{1}_{ \lbrace \theta_i < \frac{1}{3}\pi \rbrace }  \lambda_i \cos \left( \left( \frac{3}{\sqrt{2 \pi}} \theta_i \right)^2 \right)^2

ScalarNew

.. math::

    p \left( v_i \right) =  \mathbb{1}_{ \lbrace \theta_i < \frac{1}{3}\pi \rbrace }  \lambda_i \cos \left( \left( \frac{3}{\sqrt{2 \pi}} \theta_i \right)^2 \right)^2 \exp \left( - \frac{\left( \| v_i \| - \| w \| \right)^2}{\sigma} \right)

where :math:`\theta_i` denotes the angle between :math:`\pm v_i` and :math:`w` (Select :math:`\pm v_i` such
that :math:`\theta_i \leq 90`) and :math:`b` is set via :code:`-exp` and :math:`\sigma` is set via :code:`-var`.
Then the next direction is chosen by an random draw.

The output file is in ply format, which contains two elements. Firstly, vertices:
contains spatial information about the streamlines, e.g. coordinates in 3D. Further the seed-coordinate is marked by a 1.
These are saved in the properties: x, y, z, seedpoint.
Secondly, fiber. Contains the property endindex, which denotes the end index of a streamline.


bundle-filtering
~~~~~~~~~~~~~~~~
The generated streamlines can be filtered by running the following command:

.. code-block:: console

    $ bundle-filtering -i path/to/trackingResults.ply -m path/to/fODF.nrrd -o path/to/outfile.ply

If this script is applied to self generated ply data, it is important that this ply file contains the following:
Firstly, vertices:
contains spatial information about the streamlines, e.g. coordinates in 3D. Further the seed-coordinate is marked by a 1.
These are saved in the properties: x, y, z, seedpoint.
Secondly, fiber. Contains the property endindex, which denotes the end index of a streamline.

Further several filter parameters can be set:

* :code:`--mask`: Minimal streamline density. Creates a voxel mask and cuts of each streamline at the first intersection with the complement of the mask. Default 5
* :code:`--exclusion`: Filters out all streamlines which intersect with a given plane e.g. x<10. Several planes can be seperated with white spaces. Default ""
* :code:`--exclusionc`: Filters out all streamlines which intersect with a given cube e.g. 10<x<20,5<y90,40<z<100. Several cubes can be seperated by a white space Default ""
* :code:`--minlen`: Filters out all streamlines which not at least minlen long. Default 0.

Reference
----------

If you use our software as part of a scientific project, please cite the corresponding publications. The method implemented in :code:`stdeconv` and :code:`mtdeconv` was first introduced in

.. [1] Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Fast and Accurate Multi-Tissue Deconvolution Using SHORE and H-psd Tensors. In: Proc. Medical Image Analysis and Computer-Aided Intervention (MICCAI) Part III, pp. 502-510, vol. 9902 of LNCS, Springer, 2016

It was refined and extended in

.. [2] Michael Ankele, Lek-Heng Lim, Samuel Groeschel, Thomas Schultz: Versatile, Robust, and Efficient Tractography With Constrained Higher-Order Tensor fODFs. In: Int'l J. of Computer Assisted Radiology and Surgery, 12(8):1257-1270, 2017

The methods implemented in :code:`low-rank-k-approx` was first introduced in

.. [3] Thomas Schultz, Hans-Peter Seidel: Estimating Crossing Fibers: A Tensor Decomposition Approach. In: IEEE Transactions on Visualization and Computer Graphics, 14(6):1635-42, 2008

The methods implemented in :code:`peak-modelling` was first introduced in

.. [4] Johannes Grün, Gemma van der Voort, Thomas Schultz: Reducing Model Uncertainty in Crossing Fiber Tractography. In proceedings of EG Workshop on Visual Computing for Biology and Medicine, pages 55-64, 2021

Extended in:

.. [5] Johannes Grün, Gemma van der Voort, Thomas Schultz: Model Averaging and Bootstrap Consensus Based Uncertainty Reduction in Diffusion MRI Tractography. In: Computer Graphics Forum, 2022

The regularized tractography methods (joint low-rank and low-rank UKF) were first implemented in :code:`prob-tracking` and introduced in

.. [6] Johannes Grün, Samuel Gröschel, Thomas Schultz: Spatially Regularized Low-Rank Tensor Approximation for Accurate and Fast Tractography. In NeuroImage, 2023


The use of quadratic cone programming to make the kurtosis fit more stable which is implemented in :code:`kurtosis` has been explained in the methods section of

.. [7] Samuel Groeschel, G. E. Hagberg, T. Schultz, D. Z. Balla, U. Klose, T.-K. Hauser, T. Nägele, O. Bieri, T. Prasloski, A. MacKay, I. Krägeloh-Mann, K. Scheffler: Assessing white matter microstructure in brain regions with different myelin architecture using MRI. In: PLOS ONE 11(11):e0167274, 2016

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
