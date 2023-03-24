Getting Started
------------------

This introduction will help a new user to reconstruct a part of the CC tract of a HCP subject. For a more detailed description of all commands we refer to the next sections.

To follow along with this tutorial, you will require FSL. It can be downloaded via https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/ . You will also need to download HCP diffusion MRI data from the HCP website. This dataset includes high-quality diffusion MRI images and preprocessed data. You can download the data for free after registering on the HCP db website (https://db.humanconnectome.org/) and agreeing to the licence.
Please download patient 904044 by searching for 904044, clicking `download image` and selecting the `Diffusion Preprocessed`.
After downloading, please extract the files from the zip and set `HCPdir` to the root of the folder.

We first reconstruct fODFs as it was proposed by Ankele et al. [Ankele17]_. In this method the single fiber response function (which is then used for deconvolution) is estimated using voxels with a high fractional anisotropy.
To estimate the FA values we are selecting first low b-values (<1500) and the corresponding b vectors and measurements via:

.. code-block:: console

    dtiselectvols --outdata "${HCPdir}/T1w/Diffusion/dtidata.nii.gz" \
                    --outbvecs "${HCPdir}/T1w/Diffusion/dtibvecs" \
                    --outbvals "${HCPdir}/T1w/Diffusion/dtibvals" \
                    --indata "${HCPdir}/T1w/Diffusion/data.nii.gz" \
                    --inbvecs "${HCPdir}/T1w/Diffusion/bvecs"  \
                    --inbvals "${HCPdir}/T1w/Diffusion/bvals"

Next we are estimating tensors via FSL

.. code-block:: console

    dtifit -k "${HCPdir}/T1w/Diffusion/dtidata.nii.gz" \
            -o "${HCPdir}/T1w/Diffusion/dti" \
            -m "${HCPdir}/T1w/Diffusion/nodif_brain_mask.nii.gz" \
            -r "${HCPdir}/T1w/Diffusion/dtibvecs" -b "${HCPdir}/T1w/Diffusion/dtibvals" -w

The resulting file `dti_FA.nii.gz` contains the FA values and can be inspected with fsleyes.

We also require maps of white matter, gray matter, and CSF, which we obtain by segmenting the T1 image. Therefore, we apply the mask to it via

.. code-block:: console

    fslmaths "${HCPdir}/T1w/T1w_acpc_dc_restore_1.25.nii.gz"  \
                -mas "${HCPdir}/T1w/Diffusion/nodif_brain_mask.nii.gz" \
                    "${HCPdir}/T1w/Diffusion/T1_masked.nii.gz"

and run the segmentation via

.. code-block:: console

    fast -o "${HCPdir}/T1w/Diffusion/fast" "${HCPdir}/T1w/Diffusion/T1_masked.nii.gz"

Now, we have all prerequisites in place and can estimate the fODFs using

.. code-block:: console

    mtdeconv ./${HCPdir}/T1w/Diffusion -o "${HCPdir}/T1w/Diffusion/mtdeconv/" -v True

This generates multiple output files, including `fodf.nrrd` which contains the computed fODFs. If desired, it can be transformed into dipy/mrtrix format using

.. code-block:: console

    bonndit2mrtrix -i "${HCPdir}/T1w/Diffusion/mtdeconv/fodf.nrrd" \
                    -o "${HCPdir}/T1w/Diffusion/mtdeconv/fodf.nii.gz"

which will output a `fodf.nii.gz` file readable by MRtrixs' mrview etc. From the fODFs we can extract fiber orientations via low-rank approximation [Schultz08]_

.. code-block:: console

    low-rank-k-approx -i "${HCPdir}/T1w/Diffusion/mtdeconv/fodf.nrrd" \
                        -o "${HCPdir}/T1w/Diffusion/mtdeconv/rank2.nrrd" -r 2

As a final step we reconstruct a fiber bundle of the CC. Therefore, we supplied pregenerated seed points with initial directions \
in the file `bonndit/data/CC.pts`. More information about the file format is available in the documentation of :ref:`prob-tracking`.

To run the low-rank UKF, which produced fast and accurate results in [Gruen23]_, we have to use :ref:`prob-tracking` with the "ukf" flag:

.. code-block:: console

    prob-tracking -i "${HCPdir}/T1w/Diffusion/mtdeconv/" --seedpoints "test_CC" \
                    -o "cc_ukf.tck" --infile rank2.nrrd --ukf "LowRank"

To run tractography based on a joint low-rank approximation, which more fully reconstructed the fiber spread based on a single seed region in [Gruen23]_, we would prefer a rank-3 approximation

.. code-block:: console

    low-rank-k-approx -i "${HCPdir}/T1w/Diffusion/mtdeconv/fodf.nrrd" \
                        -o "${HCPdir}/T1w/Diffusion/mtdeconv/rank3.nrrd" -r 3

and run :ref:`prob-tracking` as
			
.. code-block:: console

    prob-tracking -i "${HCPdir}/T1w/Diffusion/mtdeconv/" --seedpoints "test_CC" \
                    -o "cc_joint.tck" --dist -1

Setting :code:`--dist` to a non-zero value enables the joint approximation. With :code:`--dist -1` the algorithm chooses its neighborhood automatically. Alternatively, a value > 0 includes
a specific neighborhood size. Additional control parameters for the joint low-rank approximation are :code:`--sigma_1` and :code:`--sigma_2`, which
control the weighting of the neighborhood as described in the publication.

For comparison, basic tractography without spatial regularization can be run with the following command:

.. code-block:: console

    prob-tracking -i "${HCPdir}/T1w/Diffusion/mtdeconv/" --seedpoints "test_CC" \
                    -o "cc_unregularized.tck"

All variants use iterative tractography, starting at each seed point and extending into both directions. If no direction is specified in the seed file, \
the main direction of the low-rank approximation at the closest voxel will be used. Each iteration step \
contains the following parts. First, the fODF at the current point is interpolated trilinearly from its surrounding. From the fODF we are \
calculating the low-rank approximation and choosing the next direction probabilistically. If a Runge-Kutta integration scheme is used, \
we first perform a tentative step with half the step size and redo the trilinear interpolation and direction choice to use the mean direction with full step size.
This is done until a stopping criteria is reached. By default, it is set to a minimum white matter density of 0.3 and a maximum curvature of 130 degrees over the last 30mm. More details about various options can be found under :ref:`prob-tracking`.

Streamlines can be visualized using MRtrix' `mrview`, under tools -> tractography the data can be read and will be displayed.
