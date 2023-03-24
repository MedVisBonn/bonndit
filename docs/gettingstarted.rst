Getting Started
------------------

This introduction will help a new user to reconstruct a part of the CC tract of a HCP subject. For a more detailed description of all commands we refer to the next sections.

** To run the following tutorial it is necessary to have FSL installed. It can be downloaded via https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/ **

First, you need to download HCP diffusion MRI data from the HCP website. This dataset includes high-quality diffusion MRI images and preprocessed data. You can download the data for free after registering on the HCP db website (db.humanconnectome.org/) and agreeing to the licence.
To follow the tutorial please download patient 904044. Therfore, search for 904044, click `download image` and select the `Diffusion Preporocessed`.
After downloading, please extract the files from the zip and set `HCPdir` to the root of the folder.

As an intermediate step we reconstruct fODFs as it was proposed by Ankele et al. [Ankele17]_. In this method the single fiber response function (which is then used for deconvolution) is estimated using voxels with an high fractional anisotropy.
To estimate the FA values we are selecting first low b-values (<1500) and the corresponding b vectors and measurements via:

.. code-block:: console

    dtiselectvols --outdata "${HCPdir}/T1w/Diffusion/dtidata.nii.gz" \
                    --outbvecs "${HCPdir}/T1w/Diffusion/dtibvecs" \
                    --outbvals "${HCPdir}/T1w/Diffusion/dtibvals" \
                    --indata "${HCPdir}/T1w/Diffusion/data.nii.gz" \
                    --inbvecs "${HCPdir}/T1w/Diffusion/bvecs"  \
                    --inbvals "${HCPdir}/T1w/Diffusion/bvals"

Next we are estimating tensors via FSLs'

.. code-block:: console

    dtifit -k "${HCPdir}/T1w/Diffusion/dtidata.nii.gz" \
            -o "${HCPdir}/T1w/Diffusion/dti" \
            -m "${HCPdir}/T1w/Diffusion/nodif_brain_mask.nii.gz" \
            -r "${HCPdir}/T1w/Diffusion/dtibvecs" -b "${HCPdir}/T1w/Diffusion/dtibvals" -w

The `dti_FA.nii.gz` contains the FA values and can be inspected with fsleyes.

As a second step we are segmenting the T1 image into WM/GM/CSF. Therefore, we apply the mask to it via

.. code-block:: console

    fslmaths "${HCPdir}/T1w/T1w_acpc_dc_restore_1.25.nii.gz"  \
                -mas "${HCPdir}/T1w/Diffusion/nodif_brain_mask.nii.gz" \
                    "${HCPdir}/T1w/Diffusion/T1_masked.nii.gz"

and running the segmentation via

.. code-block:: console

    fast -o "${HCPdir}/T1w/Diffusion/fast" "${HCPdir}/T1w/Diffusion/T1_masked.nii.gz"

Computed all prerequisites the fODFs can be estimated using

.. code-block:: console

    mtdeconv ./${HCPdir}/T1w/Diffusion -o "${HCPdir}/T1w/Diffusion/mtdeconv/" -v True

The `fodf.nrrd` contians the computed fODFs. It can be transformed into dipy/mrtrix format using the

.. code-block:: console

    bonndit2mrtrix -i "${HCPdir}/T1w/Diffusion/mtdeconv/fodf.nrrd" \
                    -o "${HCPdir}/T1w/Diffusion/mtdeconv/fodf.nii.gz"

command, which will output a `fodf.nii.gz` file readable by MRtrixs' mrview etc. From the fODFs we can extract fiber orientations
via [Schultz08]_

.. code-block:: console

    low-rank-k-approx -i "${HCPdir}/T1w/Diffusion/mtdeconv/fodf.nrrd" \
                        -o "${HCPdir}/T1w/Diffusion/mtdeconv/rank3.nrrd" -r 3

As a final step we reconstruct a fiber bundle of the CC. Therefore, we supplied pregenerated seed points with initial directions \
in the `bonndit/data/CC.pts` file. For more information about the file format have a look into the tracking section.

To run the easiest version of the tractography code we run the following command:

.. code-block:: console

    prob-tracking -i "${HCPdir}/T1w/Diffusion/mtdeconv/" --seedpoints "test_CC" \
                    -o "cst_unconstrained.tck"

It uses an iterative tractography approach beginning at each seed point into both directions. If no direction is specified in the seed file it will \
use the main direction of low-rank approximation at the closest voxel. Now it will track iteratively into both directions. Each iteration steps \
contains the following parts. First the fODF at the current point is interpolated trilinearly from its surrounding. From the fODF we are \
calculating the low-rank approximation [Gruen23]_ and choosing the next direction probabilistically. Using a Runge-Kutta integration scheme \
we are doing a step with half step size and redo the trilinear interpolation and direction choice to use the mean direction with full step size.
This is done until a stopping criteria is reached, which are set to a minimum wm density of 0.3 and a maximum curvature of 130 degrees over the last 30mm.

To run the more advanced joint low-rank approximation we have to specify

.. code-block:: console

    prob-tracking -i "${HCPdir}/T1w/Diffusion/mtdeconv/" --seedpoints "test_CC" \
                    -o "cst_constrained.tck"

Instead of using the low-rank approximation, we are using a regularised version of it the joint low-rank approximation, which was introduced in [5]_ \
as first method.

To run the low-rank UKF we have to add the "ukf" flag.

.. code-block:: console

    prob-tracking -i "${HCPdir}/T1w/Diffusion/mtdeconv/" --seedpoints "test_CC" \
                    -o "cst_ukf.tck" --ukf "LowRank"

We have replaced the low-rank approximation with an UKF approach which estimated the new low-rank approximation depending on the past and regularize \
through this. This was introduced in [Gruen23]_ as second approach.

Streamlines can be visualized using MRtrix' `mrview`, under tools -> tractography the data can be read and will be displayed.

More details about various options can be found below.