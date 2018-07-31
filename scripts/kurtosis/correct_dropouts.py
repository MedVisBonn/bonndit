from __future__ import division
import models
import time
import pandas as pd
import os
import numpy as np
import dipy
import nibabel as nib
from dipy.io import read_bvals_bvecs
import configparser
import argparse

parser = argparse.ArgumentParser(description = 'Models to detect and impute signal outliers')
parser.add_argument('datafile', help='Path to the DWI data file')
parser.add_argument('bvecfile', help='Path to the normalized gradient directions (b-vectors) file')
parser.add_argument('bvalfile', help='Path to the b-values file')
parser.add_argument('model',choices=['REKINDLE', 'IRL1SHORE', 'relSHORE'] ,help='Choice of the kurtosis model')
parser.add_argument('-m', '--mask', help='Path to mask file')
parser.add_argument('-v', '--verbose', action='store_true', help='Flag for verbose output')
parser.add_argument('-c', '--configfile', help='Path to config .ini file to set all model parameters')
args = parser.parse_args()


# Loading data
def load_data(datafile):
    '''
    Loads the data from a -nii data file
    :param datafile:
    :return: data, affine
    '''
    img = nib.load(datafile)
    data = img.get_data()
    data = np.array(data)
    affine = img.affine
    return data, affine


data_file = args.datafile
dir_path = os.path.dirname(os.path.realpath(data_file))
data, affine = load_data(data_file)

bvals, bvecs = dipy.io.read_bvals_bvecs(args.bvalfile, args.bvecfile)

model = args.model
mask = args.mask
verbose = args.verbose

if mask:
    mask = load_data(mask)[0]

if args.configfile is None:
    if os.path.exists(dir_path + '/' + 'parameters.ini'):
        parameters_file = dir_path + '/' + 'parameters.ini'
    else:
        models.create_parameters_file(dir_path)
        parameters_file = dir_path + '/' + 'parameters.ini'

else:
    parameters_file = args.configfile

config = configparser.ConfigParser()
config.read(parameters_file)

data_file = data_file.replace(dir_path, '').strip('/')


def unmask(array, mask):
    '''
    Unmasking an array by the mask array
    :param masked array:
    :param mask:
    :return: unmasked array
    '''
    tmp = np.zeros((mask.shape[0], array.shape[1]), dtype=array.dtype)
    tmp[mask] = array
    return tmp


def rekindle():
    '''
    Main function to execute the REKINDLE algorithm
    :return: output files as specified by the user
    '''
    parameters = config['REKINDLE']

    X = models.design_matrix(bvals, bvecs, affine)
    data_reshaped = data.reshape((-1, data.shape[3]))

    if mask is not None:
        mask_reshaped = mask.flatten()
        mask_reshaped = mask_reshaped == 1
        data_masked = data_reshaped[mask_reshaped]
    else:
        data_masked = data_reshaped

    # Creating numpy arrays to hold the output data
    result_betas = np.zeros((data_masked.shape[0], X.shape[1]), np.dtype(float))
    result_inliers = np.zeros(data_masked.shape, dtype=np.bool)
    result_residuals = np.zeros(data_masked.shape, np.dtype(float))
    result_predictions = np.zeros(data_masked.shape, np.dtype(float))
    num_outliers = 0
    skipped_voxels = 0

    start = time.time()
    for i in range(data_masked.shape[0]):
        model_ = models.REKINDLE(parameters.getfloat('regularization_constant'), parameters.getfloat('kappa'),
                                 parameters.getfloat('c'), parameters.getint('irls_maxiter'),
                                 parameters.getint('rekindle_maxiter'), verbose)
        residuals = model_.outlier_scoring(X, data_masked[i])
        inliers = model_.outlier_detection(residuals)[0]
        outliers = model_.outlier_detection(residuals)[1]
        betas = model_.fit(X[inliers], data_masked[i][inliers])
        prediction = model_.predict(X)

        if (i % 1000) == 0:
            print('\t{:.2%} Completed, Time elapsed: {}'.format(i/data_masked.shape[0], int(time.time()-start)))

        result_residuals[i] = residuals
        result_inliers[i] = inliers
        result_betas[i] = betas.flatten()
        result_predictions[i] = prediction
        num_outliers += np.count_nonzero(outliers)

    if mask is not None:
        result_betas_unmasked = unmask(result_betas, mask_reshaped)
        result_inliers_unmasked = unmask(result_inliers, mask_reshaped)
        result_residuals_unmasked = unmask(result_residuals, mask_reshaped)
        result_predictions_unmasked = unmask(result_predictions, mask_reshaped)

    else:
        result_betas_unmasked = result_betas
        result_inliers_unmasked = result_inliers
        result_residuals_unmasked = result_residuals
        result_predictions_unmasked = result_predictions

    predictions = pd.DataFrame(result_predictions_unmasked)
    inliers = pd.DataFrame(result_inliers_unmasked)

    if parameters['imputation'] == 'replace_outliers':
        correct_data = pd.DataFrame(data_reshaped)
        correct_data = correct_data[inliers]
        correct_data[correct_data.isnull()] = predictions
        skipped_voxels += correct_data.isnull().sum().sum()
        if parameters.getboolean('pickle_files'):
            pd.to_pickle(correct_data, os.path.join(dir_path, 'output_'+model + '_' +
                                                    data_file.rstrip('.nii') + '_corrected_data.pkl'))
        final_data = correct_data.values.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        correct_data_image = nib.Nifti1Image(final_data, affine)
        nib.save(correct_data_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                  '_corrected_data_image.nii'))

    elif parameters['imputation'] == 'replace_all':
        skipped_voxels += predictions.isnull().sum().sum()
        final_predictions = result_predictions_unmasked.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        predictions_image = nib.Nifti1Image(final_predictions, affine)
        nib.save(predictions_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                 '_predictions_image.nii'))
        if parameters.getboolean('pickle_files'):
            pd.to_pickle(predictions, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                   '_predictions.pkl'))

    if parameters.getboolean('score_file'):
        final_residuals = result_residuals_unmasked.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        residuals_image = nib.Nifti1Image(final_residuals, affine)
        nib.save(residuals_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                               '_residuals_image.nii'))
        if parameters.getboolean('pickle_files'):
            residuals = pd.DataFrame(result_residuals_unmasked)
            pd.to_pickle(residuals, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                 '_residuals_score.pkl'))

    if parameters.getboolean('beta_file'):
        final_betas = result_betas_unmasked.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        betas_image = nib.Nifti1Image(final_betas, affine)
        nib.save(betas_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                           '_betas_image.nii'))
        if parameters.getboolean('pickle_files'):
            betas = pd.DataFrame(result_betas_unmasked)
            pd.to_pickle(betas, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                             '_betas.pkl'))

    if parameters.getboolean('log_file'):
        with open(os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') + '_log_file.txt'), 'w') \
                  as f:
            f.write('Model used to detect and impute signal outliers: ' + model + 2*'\n')
            f.write('Path to the DWI data file: ' + args.datafile + '\n')
            f.write('Path to the normalized gradient directions (b-vectors) file: ' + args.bvecfile + '\n')
            f.write('Path to the b-values file: ' + args.bvalfile + 2*'\n')
            f.write('Algorithm parameters:' + 2*'\n')
            for p in parameters:
                f.write(p + ': ' + parameters[p] + '\n')
            f.write('mask_file: ' + str(args.mask) + 2*'\n')
            f.write('Number of outliers in the dataset: ' + str(num_outliers) + '\n')
            f.write('Ratio of outliers in the dataset: ' + str(round(num_outliers / data_reshaped.size,5)) + '\n')
            f.write('Number of skipped voxels: ' + str(skipped_voxels))


def irl1shore(relshore=False):
    '''
    Main function to execute the SHORE model to correct and impute signal outliers
    :param relshore:
    :return: output files as specified by the user
    '''
    if relshore:
        parameters = config['relSHORE']
    else:
        parameters = config['IRL1SHORE']

    radial_order = parameters.getint('radial_order')
    zeta = parameters.getint('zeta')
    tau = parameters.getfloat('tau')

    X = models.shore_matrix(bvals, bvecs, radial_order, zeta, tau)

    data_reshaped = data.reshape((-1, data.shape[3]))

    if mask is not None:
        mask_reshaped = mask.flatten()
        mask_reshaped = mask_reshaped == 1
        data_masked = data_reshaped[mask_reshaped]
    else:
        data_masked = data_reshaped

    result_betas = np.zeros((data_masked.shape[0], X.shape[1]), np.dtype(float))
    result_residuals = np.zeros(data_masked.shape, np.dtype(float))
    result_inliers = np.zeros(data_masked.shape, dtype=np.bool)
    result_predictions = np.zeros(data_masked.shape, np.dtype(float))
    num_outliers = 0
    skipped_voxels = 0

    start = time.time()
    for i in range(data_masked.shape[0]):
        y = data_masked[i]
        if relshore:
            model_ = models.IRLSHORE(parameters.getfloat('lambda'),
                                     0,
                                     'relative',
                                     parameters.getfloat('t_threshold'),
                                     parameters.getfloat('power_relative'),
                                     False,
                                     parameters.getboolean('one_sided_scoring'),
                                     0,
                                     radial_order,
                                     zeta,
                                     tau,
                                     verbose,
                                     S0=np.mean(y[bvals == 0]))
        else:
            model_ = models.IRLSHORE(parameters.getfloat('lambda'),
                                     parameters.getint('max_iter'),
                                     parameters['residual_calculation'],
                                     parameters.getfloat('t_threshold'),
                                     0,
                                     parameters.getboolean('one_sided_reweighting'),
                                     parameters.getboolean('one_sided_scoring'),
                                     parameters.getfloat('convergence_threshold'),
                                     radial_order,
                                     zeta,
                                     tau,
                                     verbose,
                                     S0=np.mean(y[bvals == 0]))

        residuals = model_.outlier_scoring(X, y)
        outliers = model_.outlier_detection(residuals)
        inliers = (~outliers)
        betas = model_.fit(X[inliers], y[inliers])
        predictions = model_.predict(X, y[inliers])

        if (i % 1000) == 0:
            print('\t{:.2%} Completed, Time elapsed: {}'.format(i/data_masked.shape[0], int(time.time()-start)))

        result_residuals[i] = residuals
        result_inliers[i] = inliers
        result_betas[i] = betas
        result_predictions[i] = predictions
        num_outliers += np.count_nonzero(outliers)

    if mask is not None:
        result_residuals_unmasked = unmask(result_residuals, mask_reshaped)
        result_inliers_unmasked = unmask(result_inliers, mask_reshaped)
        result_betas_unmasked = unmask(result_betas, mask_reshaped)
        result_predictions_unmasked = unmask(result_predictions, mask_reshaped)
    else:
        result_residuals_unmasked = result_residuals
        result_inliers_unmasked = result_inliers
        result_betas_unmasked = result_betas
        result_predictions_unmasked = result_predictions

    predictions = pd.DataFrame(result_predictions_unmasked)
    inliers = pd.DataFrame(result_inliers_unmasked)

    if parameters['imputation'] == 'replace_outliers':
        correct_data = pd.DataFrame(data_reshaped)
        correct_data = correct_data[inliers]
        correct_data[correct_data.isnull()] = predictions
        skipped_voxels += correct_data.isnull().sum().sum()
        if parameters.getboolean('pickle_files'):
            pd.to_pickle(correct_data, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                    '_corrected_data.pkl'))
        final_data = correct_data.values.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        correct_data_image = nib.Nifti1Image(final_data, affine)
        nib.save(correct_data_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                  '_corrected_data_image.nii'))

    elif parameters['imputation'] == 'replace_all':
        skipped_voxels += predictions.isnull().sum().sum()
        final_predictions = result_predictions_unmasked.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        predictions_image = nib.Nifti1Image(final_predictions, affine)
        nib.save(predictions_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                 '_predictions_image.nii'))
        if parameters.getboolean('pickle_files'):
            pd.to_pickle(predictions, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                   '_predictions.pkl'))

    if parameters.getboolean('score_file'):
        final_residuals = result_residuals_unmasked.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        residuals_image = nib.Nifti1Image(final_residuals, affine)
        nib.save(residuals_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                               '_residuals_image.nii'))
        if parameters.getboolean('pickle_files'):
            residuals = pd.DataFrame(result_residuals_unmasked)
            pd.to_pickle(residuals, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                                 '_residuals_score.pkl'))

    if parameters.getboolean('beta_file'):
        final_betas = result_betas_unmasked.reshape((data.shape[0], data.shape[1], data.shape[2], -1))
        betas_image = nib.Nifti1Image(final_betas, affine)
        nib.save(betas_image, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                           '_betas_image.nii'))
        if parameters.getboolean('pickle_files'):
            betas = pd.DataFrame(result_betas_unmasked)
            pd.to_pickle(betas, os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') +
                                             '_betas.pkl'))

    if parameters.getboolean('log_file'):
        with open(os.path.join(dir_path, 'output_' + model + '_' + data_file.rstrip('.nii') + '_log_file.txt'), 'w') \
                  as f:
            f.write('Model used to detect and impute signal outliers: ' + model + 2*'\n')
            f.write('Path to the DWI data file: ' + args.datafile + '\n')
            f.write('Path to the normalized gradient directions (b-vectors) file: ' + args.bvecfile + '\n')
            f.write('Path to the b-values file: ' + args.bvalfile + 2*'\n')
            f.write('Algorithm parameters:' + 2*'\n')
            for p in parameters:
                f.write(p + ': ' + parameters[p] + '\n')
            f.write('mask_file: ' + str(args.mask) + 2*'\n')
            f.write('Number of outliers in the dataset: ' + str(num_outliers) + '\n')
            f.write('Ratio of outliers in the dataset: ' + str(round(num_outliers / data_reshaped.size, 5)) + '\n')
            f.write('Number of skipped voxels: ' + str(skipped_voxels))


if model == 'REKINDLE':
    rekindle()
elif model == 'IRL1SHORE':
    irl1shore()
elif model == 'relSHORE':
    irl1shore(relshore=True)
