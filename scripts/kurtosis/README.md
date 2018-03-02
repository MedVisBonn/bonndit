# An implementation of the kurtosis tensor model 

This implementation consists of two scripts. `kurtosis-cone.py` is used to fit the parameters of the kurtosis model
similar to [Tabesh et al., MRM 2011](https://doi.org/10.1002/mrm.22932). It guarantees minimum diffusivity by using cone
programming and uses the positive definiteness constraint (H-psd) described in
[Ankele et al., 2017](https://doi.org/10.1007/s11548-017-1593-6). The script yields a parameter file with the following 
contents:

+ A mask which is 1 for all voxels with meaningful data and 0 for the rest
+ The 6 elements of the diffusion tensor (DT) in lexicographical order (xx, xy, xz, yy, yz, zz)
+ The 15 elements of the kurtosis tensor (KT) in lexicographical order (xxxx, xxxy, xxxz, xxyy, xxyz, ...)

The script `kurtosis-measures.py` uses the parameter file to derive and store the following measures:

+ Axial diffusivity (_DA.nii)
+ Radial diffusivity (_DR.nii)
+ Mean diffusivity (DM.nii)
+ Fractional anisotropy (_FA.nii)
+ Axial kurtosis (_KA.nii)
+ Radial kurtosis (_KR.nii)
+ Mean kurtosis (_KM.nii)

The computation of axial, radial, and mean kurtosis from the DKI parameters is based on the equations in 
[Tabesh et al., MRM 2011](https://doi.org/10.1002/mrm.22932).

## Getting Started

If you want to use the scripts for your diffusion weighted data. you need to clone this repository and install the 
required packages. Than you navigate to the folder named `scripts/` and run the scripts from the command line. A listing
and description of all possible parameter is shown to you if you use the `-h` parameter:

```
./kurtosis-cone.py -h
```

### Prerequisites

All requirements can be installed via pip from the requirements.txt file:

```
pip install -r requirements.txt
```


### Installing

Complete installation via pip will be available soon.

## Authors

* **Thomas Schulz** - *Initial work* - [github_account](https://github.com/username)

* **Michael Ankele** - *Initial work* - [github_account](https://github.com/username)

* **Maybe some others** - *Initial work* - [github_account](https://github.com/username)

See also the list of [contributors](https://github.com/project/contributors) who participated in this project.

## License

This project is licensed under **some license** - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* ...