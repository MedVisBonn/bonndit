# Multi-shell Multi-Tissue deconvolution

Spherical deconvolution for diffusion MRI data with multiple b values. Includes a re-implementation of the approach by Jeurissen et al., which assumes data on spherical shells, and our own approach (MICCAI 2016), which uses SHORE to accept either shells or random sampling, DSI, etc.

## How to use

Since a large number of input files are required and we don't want to specify each of them on the command line, the expected structure consists of the following files in a single input folder:

bvals / bvecs - b values and vectors in FSL format (ASCII files)
data.nii(.gz) - The actual 4D DWI data in FSL format
dti_FA.nii.gz - FA maps derived from the data (e.g., using FSL)
dti_V1.nii.gz - Principal eigenvectors from the data (e.g., using FSL)
fast_pve_{0,1,2}.nii.gz - Maps of CSF, GM, WM, respectively
                          Can be obtained from anatomical T1 images using
			  tissue segmentation (e.g., using FSL's fast)
			  Have to be warped to the same space as DWI data
mask.nii.gz   - brain mask (optional, but recommended)

Both approaches first need to estimate the response functions, e.g.

shore-response.py input-dir/ output-dir/

The responses and masks of the voxels from which they were computed get written into output-dir/

Afterwards, the actual deconvolution can be computed, e.g.

shore-deconvolve.py input-dir/ output-dir/ --constrainPos posdef

Different constraints can be selected; posdef is the H-psd constraint proposed in our MICCAI paper and works best in practice.