
"""
Created on Tue Jun  5 16:36:15 2018

@author: mahgoub
"""

from __future__ import division
import time
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
import dipy
import nibabel as nib
from dipy.io import read_bvals_bvecs
import configparser
import argparse
from dipy.reconst.shore import shore_matrix
from dipy.core.gradients import gradient_table
import sklearn
from sklearn import linear_model




parser = argparse.ArgumentParser()
parser.add_argument('datafile', help='Path to data file')
parser.add_argument('bvecfile', help='Path to bvec file')
parser.add_argument('bvalfile', help='Path to bval file')
parser.add_argument('model',choices=['REKINDLE', 'IRL1SHORE', 'relSHORE'] ,help='Choice of the kurtosis model')
parser.add_argument('--mask', help='Path to mask file')
parser.add_argument('--configfile', help='Config file to set all model parameters as a .ini file')
args = parser.parse_args()


# Loading data
data_file = args.datafile
img    = nib.load(data_file)
data   = img.get_data()
data   = np.array(data)
affine = img.affine

bvals, bvecs = dipy.io.read_bvals_bvecs(args.bvalfile,args.bvecfile)

model  = args.model
mask   = args.mask

if args.configfile == None:
    parameters_file = 'parameters.ini'
else:
    parameters_file = args.configfile

config = configparser.ConfigParser()
config.read(parameters_file)

def unmask(array, mask):
    tmp = np.zeros((mask.shape[0], array.shape[1]), dtype=array.dtype)
    tmp[mask] = array
    return tmp
    
MAD = lambda residuals: np.median(np.abs(residuals - np.median(residuals, axis=0)), axis = 0)   #Median absolute deviation given 1d array

def LLSfit(X, y, regularization_constant):
    return np.linalg.inv(X.transpose() * X + np.matrix(np.eye(X.shape[1])) * regularization_constant) * X.transpose() *     y #fits Linear Least Squares
def WLLSfit(X, y, W, regularization_constant):
    return np.linalg.inv(X.transpose() * W * X + np.matrix(np.eye(X.shape[1])) * regularization_constant) * X.transpose() * W * y #fits Weighted Linear Least Squares give weights

class REKINDLE(object):
    def __init__(self,
                 regularization_constant,
                 kappa,
                 c,
                 IRLS_maxiter,
                 REKINDLE_maxiter,
                 verbose):
        self.regularization_constant = regularization_constant #Prevents matrix inversion from failure when inverting degenerate matrices
        self.kappa = kappa
        self.c = c
        self.IRLS_maxiter = IRLS_maxiter
        self.REKINDLE_maxiter = REKINDLE_maxiter
        self.verbose = verbose

    def conv_check(self, beta1, beta2):
        return (np.abs(beta1-beta2) - self.c* np.max(np.concatenate((np.abs(beta1), np.abs(beta2)), axis= 1), axis=1)).max()        #Calculates convergence criterion (<0 is good)

    def IRLS__(self, X, y, beta_init, weighting ='geman-mcclure'):  #Iteratively Reweighted Least Squares (linear) weighting parameter according to weighting used
        beta_old = beta_init

        counter = 0
        tag =True
        
        while tag:       
            if  weighting == 'geman-mcclure':
                residuals              = np.array(y - X * beta_old)
                residuals_sigma        = 1.4826 * MAD(residuals)
                residuals_standardized = residuals / residuals_sigma
                weights = 1.0/(residuals_standardized**2 + 1)**2
            elif weighting =='exponential':
                weights = np.exp(2* np.array(X*beta_old))

            W = np.matrix(np.eye(X.shape[0]) * weights)

            beta_new = WLLSfit(X, y, W, self.regularization_constant)

    #         convergence_check = (np.abs(beta_new-beta_old) - c* np.max(np.concatenate((np.abs(beta_new), np.abs(beta_old)), axis= 1), axis=1)).max()
            convergence_check = self.conv_check(beta_new, beta_old)

            if self.verbose:
                print('Iteration {:03}. Convergence criterion: {:.2e}'.format(counter+1, convergence_check))
            
            if (convergence_check>0):
                beta_old = beta_new.copy()
                counter+=1
            else:
                tag = False
                if self.verbose:
                    print('Exiting due to convergence.')

            if counter>self.IRLS_maxiter:
                tag = False
                if self.verbose:
                    print('Exiting due to maximum number of iterations reached. ')        
        return beta_new
    
    def signal_preprocessing(self, y):
        assert len(y.shape) == 1
        return np.log(y)
    def signal_unpreprocessing(self, y):
        assert len(y.shape) == 1
        return np.exp(y)

    def outlier_scoring(self, X, y):   
        s = self.signal_preprocessing(y)
        # Removing impossible values from further fit
        inliers = ~np.isnan(s)
        s = np.matrix(s).transpose()
#         inliers[y < 1e-10] = False  
        tag = True
        counter= 0 
        beta_old = LLSfit(X[inliers], s[inliers], self.regularization_constant) #np.linalg.inv(X.transpose() * X) * X.transpose() * y
        
        while tag:
            beta_robust   = self.IRLS__(X[inliers], s[inliers], beta_old)
            s_star = s / np.exp(-X * beta_robust)
            X_star = X / np.exp(-X * beta_robust)

            beta_rescaled = LLSfit(X_star[inliers], s_star[inliers], self.regularization_constant)
            beta_new      = self.IRLS__(X_star[inliers], s_star[inliers], beta_rescaled)

            convergence_check = self.conv_check(beta_new, beta_old)
            if self.verbose:
                    print('Iteration {:03}. Convergence criterion: {:.2e}'.format(counter+1, convergence_check))
            if (convergence_check>0):
                beta_old = beta_new.copy()
                counter+=1
            else:
                if self.verbose:
                    print('Exiting due to convergence.')
                tag = False
            if counter>self.REKINDLE_maxiter:
                tag = False
                if self.verbose:
                    print('Exiting due to maximum number of iterations reached. ')

        residuals_star = s_star - X_star* beta_new
        residuals_sigma        = 1.4826 * MAD(residuals_star[inliers])
        residuals_normalized = residuals_star.flatten()/residuals_sigma
        return residuals_normalized
    
    def outlier_detection(self, residuals_normalized): 
        inliers = (np.array(np.abs(residuals_normalized) < self.kappa).ravel())
        outliers = ~(inliers) 
        return tuple((inliers,outliers))

    def fit(self, X, y):
        s = self.signal_preprocessing(y)
        s = np.matrix(s).transpose()
        self.beta = self.IRLS__(X, s, LLSfit(X, s, self.regularization_constant), weighting = 'exponential')
        return self 
    
    def predict(self, X):
        #beta = beta.reshape((22,1))
        predict_matrix = X * self.beta
        return self.signal_unpreprocessing(np.array(predict_matrix).flatten())

def design_matrix(bvals, bvecs, affine):

        # re-scale b values to more natural units
    bvals /= 1000.0

        # flip the signs of bvec coords as needed
    for i in range(3):
        if affine[i,i]<0:
            bvecs[:,i]=-bvecs[:,i]

    X = np.zeros((bvals.shape[0], 22), dtype=np.float64)
    X[:,  0] = 1.0
    X[:,  1]=- bvals             *      bvecs[:, 0] * bvecs[:, 0]
    X[:,  2]=- bvals             *  2 * bvecs[:, 0] * bvecs[:, 1]
    X[:,  3]=- bvals             *  2 * bvecs[:, 0] * bvecs[:, 2]
    X[:,  4]=- bvals             *      bvecs[:, 1] * bvecs[:, 1]
    X[:,  5]=- bvals             *  2 * bvecs[:, 1] * bvecs[:, 2]
    X[:,  6]=- bvals             *      bvecs[:, 2] * bvecs[:, 2]
    X[:,  7]= (bvals ** 2) / 6.0 *      bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 0]
    X[:,  8]= (bvals ** 2) / 6.0 *  4 * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 1]
    X[:,  9]= (bvals ** 2) / 6.0 *  4 * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 2]
    X[:, 10]= (bvals ** 2) / 6.0 *  6 * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 1] * bvecs[:, 1]
    X[:, 11]= (bvals ** 2) / 6.0 * 12 * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 1] * bvecs[:, 2]
    X[:, 11]= (bvals ** 2) / 6.0 *  6 * bvecs[:, 0] * bvecs[:, 0] * bvecs[:, 2] * bvecs[:, 2]
    X[:, 13]= (bvals ** 2) / 6.0 *  4 * bvecs[:, 0] * bvecs[:, 1] * bvecs[:, 1] * bvecs[:, 1]
    X[:, 14]= (bvals ** 2) / 6.0 * 12 * bvecs[:, 0] * bvecs[:, 1] * bvecs[:, 1] * bvecs[:, 2]
    X[:, 15]= (bvals ** 2) / 6.0 * 12 * bvecs[:, 0] * bvecs[:, 1] * bvecs[:, 2] * bvecs[:, 2]
    X[:, 16]= (bvals ** 2) / 6.0 *  4 * bvecs[:, 0] * bvecs[:, 2] * bvecs[:, 2] * bvecs[:, 2]
    X[:, 17]= (bvals ** 2) / 6.0 *      bvecs[:, 1] * bvecs[:, 1] * bvecs[:, 1] * bvecs[:, 1]
    X[:, 18]= (bvals ** 2) / 6.0 *  4 * bvecs[:, 1] * bvecs[:, 1] * bvecs[:, 1] * bvecs[:, 2]
    X[:, 19]= (bvals ** 2) / 6.0 *  6 * bvecs[:, 1] * bvecs[:, 1] * bvecs[:, 2] * bvecs[:, 2]
    X[:, 20]= (bvals ** 2) / 6.0 *  4 * bvecs[:, 1] * bvecs[:, 2] * bvecs[:, 2] * bvecs[:, 2]
    X[:, 21]= (bvals ** 2) / 6.0 *      bvecs[:, 2] * bvecs[:, 2] * bvecs[:, 2] * bvecs[:, 2]
    X = np.matrix(X)
    return X
        
def rekindle():
    parameters = config['REKINDLE']
    
    X = design_matrix(bvals, bvecs, affine)
    data_reshaped = data.reshape((-1,data.shape[3]))    
    
    if mask is not None:
        mask_reshaped = mask.flatten()
        mask_reshaped = mask_reshaped ==1
        data_masked   = data_reshaped[mask_reshaped]
    else:
        data_masked = data_reshaped
    
    #result_betas = np.zeros((data_masked.shape[0], X.shape[1]), dtype=np.float64)
    result_outliers = np.zeros(data_masked.shape, dtype=np.bool)
    result_inliers = np.zeros(data_masked.shape, dtype=np.bool)
    result_residuals = np.zeros(data_masked.shape, dtype=np.float64)
    result_predictions = np.zeros(data_masked.shape, dtype=np.float64)
    
    for i in range(data_masked.shape[0]):
        model_ = REKINDLE(parameters.getfloat('regularization_constant'),parameters.getfloat('kappa'),parameters.getfloat('c'),
                          parameters.getint('irls_maxiter'),parameters.getint('rekindle_maxiter'),parameters.getboolean('verbose'))
        residuals = model_.outlier_scoring(X, data_masked[i])
        inliers = model_.outlier_detection(residuals)[0]
        outliers = model_.outlier_detection(residuals)[1]
        model_.fit(X[inliers], data_masked[i][inliers])
        prediction = model_.predict(X)
    
        result_residuals[i] = residuals
        result_outliers[i] = outliers
        result_inliers[i] = inliers
        #result_betas[i] = beta
        result_predictions[i] = prediction
    
    
    if mask is not None:
        #result_betas_unmasked     = unmask(result_betas    , mask_reshaped)
        result_outliers_unmasked   = unmask(result_outliers  , mask_reshaped)
        result_inliers_unmasked   = unmask(result_inliers  , mask_reshaped)
        result_residuals_unmasked = unmask(result_residuals, mask_reshaped)
        result_predictions =        unmask(result_predictions, mask_reshaped)

    else:
        #result_betas_unmasked     = result_betas
        result_outliers_unmasked   = result_outliers
        result_inliers_unmasked   = result_inliers
        result_residuals_unmasked = result_residuals
        result_predictions_unmasked = result_predictions
    
    
    predictions = pd.DataFrame(result_predictions_unmasked)
    inliers     = pd.DataFrame(result_inliers_unmasked)
    
    if parameters['imputation'] == 'replace_outliers':
        correct_data = pd.DataFrame(data_reshaped)
        correct_data = correct_data[inliers]
        correct_data[correct_data.isnull()] = predictions
        pd.to_pickle(correct_data, 'output_REKINDLE_' + data_file.rstrip('.nii') + '_corrected_data.pkl')
       
    elif parameters['imputation'] == 'replace_all':
        pd.to_pickle(predictions, 'output_REKINDLE_' + data_file.rstrip('.nii') + '_predictions.pkl') 
        
    if parameters.getboolean('score_file'):
        residuals = pd.DataFrame(result_residuals_unmasked)
        pd.to_pickle(residuals, 'output_REKINDLE_' + data_file.rstrip('.nii') + '_residuals_score.pkl')
        

class IRLSHORE(object):
    def __init__(self, 
                 Lambda,
                 max_iter,
                 residual_calculation, 
                 verbose,
                 T_threshold,
                 power_relative, 
                 one_sided_reweighting, 
                 one_sided_scoring, 
                 convergence_threshold,
                 radial_order,
                 zeta,
                 tau,
                 S0 = 0):
        self.max_iter = max_iter
        self.Lambda = Lambda
        self.verbose = verbose
        self.S0 = S0
        self.T_threshold = T_threshold
        self.one_sided_reweighting = one_sided_reweighting
        self.one_sided_scoring = one_sided_scoring
        self.convergence_threshold = convergence_threshold
        self.radial_order = radial_order
        self.zeta = zeta
        self.tau = tau
        if Lambda == 0:
            self.regressor = linear_model.LinearRegression()
        else:
            self.regressor = linear_model.Lasso(alpha = Lambda)
        
        if   residual_calculation == 'REKINDLE':
            self.compute_residuals = self.compute_residuals_REKINDLE
        elif residual_calculation == 'relative':
            self.power_relative = power_relative
            self.compute_residuals = self.compute_residuals_relative
        elif residual_calculation == 'REKINDLE_log':
            self.compute_residuals = self.compute_residuals_REKINDLE_log
        elif residual_calculation == 'SHORE':
            self.compute_residuals = self.compute_residuals_SHORE
        elif residual_calculation == 'SHORE_log':
            self.compute_residuals = self.compute_residuals_SHORE_log
            

    def signal_preprocessing(self, y):
        return y / self.S0
    def signal_unpreprocessing(self, y):
        return y * self.S0
    

    def compute_residuals_REKINDLE(self, s_true, s_pred, inliers, one_sided = False):
        residuals = s_true - s_pred
        residuals_sigma        = 1.4826 * MAD(residuals[inliers])
        residuals_standardized = residuals / residuals_sigma
        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized
    def compute_residuals_relative(self, s_true, s_pred, inliers, one_sided = False):

        residuals = (s_true - s_pred)/(np.clip(s_pred, 1e-3, np.inf) ** self.power_relative)
        residuals_sigma        = 1.4826 * MAD(residuals[inliers])
        residuals_standardized = residuals  / residuals_sigma
        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized
    def compute_residuals_REKINDLE_log(self, s_true, s_pred, inliers, one_sided = False):
        S_true = self.signal_unpreprocessing(s_true)
        S_log = np.log(S_true)
        s_pred[s_pred < 1e-3] = 1e-3

        S_pred_log = np.log(signal_unpreprocessing(s_pred))

        residuals = S_log - S_pred_log
        residuals_sigma        = 1.4826 * MAD(residuals[inliers])       
        residuals_standardized = residuals / residuals_sigma

        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized

    def compute_residuals_SHORE(self, s_true, s_pred, inliers, one_sided = False):
        residuals       = s_true - s_pred
        residuals_sigma = 1.4826 * MAD(residuals[inliers])       
        s0_pred = self.regressor.predict(
            shore_matrix(np.array(0, ndmin = 1), np.array([0,0,0], ndmin=2),self.radial_order, self.zeta, self.tau))

        residuals_standardized = (s0_pred * s_true - s_pred)/ (residuals_sigma * ((s_true**2 + 1)**0.5))
        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized
   
    def compute_residuals_SHORE_log(self, s_true, s_pred, inliers, one_sided = False):
        S_true = self.signal_unpreprocessing(s_true)
        S_log = np.log(S_true)
        s_pred[s_pred < 1e-3] = 1e-3
        S_hat_log = np.log(self.signal_unpreprocessing(s_pred))   #problem

        residuals = S_log - S_hat_log
        residuals_sigma        = 1.4826 * MAD(residuals[inliers])           
        residuals_standardized = (S_log - S_hat_log)/ (residuals_sigma * ((S_log**2 + 1)**0.5))

        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized


    def outlier_scoring(self, X, y):
        inliers = np.ones((y.shape[0]), dtype=np.bool)
        inliers[y < 1e-10] = False
        if ~np.any(inliers):
            return -np.inf * np.ones(y.shape)
        #Normalizing signal
        s = self.signal_preprocessing(y)
        self.regressor.fit(X[inliers], s[inliers])    
        beta_old = self.regressor.coef_.copy()

        Omega_old = np.zeros((y.shape))        
        iter_count = 0
        while(True):
            iter_count +=1
            if iter_count >= self.max_iter:
                beta_new = self.regressor.coef_.copy()
                if self.verbose:
                    print('Exiting due to maximum number of iterations reached')           
                break
            s_hat = self.regressor.predict(X)
            residuals_standardized = self.compute_residuals(s, s_hat, inliers, one_sided= self.one_sided_reweighting)
            Omega   = 1.0/(residuals_standardized**2 + 1) 
            assert ~np.isnan(Omega).any()
            self.regressor.fit(np.dot(np.diag(Omega[inliers]), X[inliers]), Omega[inliers]* s[inliers])
            
            beta_new = self.regressor.coef_.copy()
              #convergence_check = np.linalg.norm(beta_new-beta_old, ord = 2)       #Original proposition
            convergence_check = np.linalg.norm(Omega - Omega_old, ord = 2)

            if convergence_check<self.convergence_threshold:
                if self.verbose:
                    print('Exiting due to convergence')
                break        
            Omega_old = Omega.copy()
            beta_old = beta_new
            
        s_hat = self.regressor.predict(X)
        residuals_standardized = self.compute_residuals(s, s_hat, inliers, one_sided= self.one_sided_scoring)
        return residuals_standardized

    def outlier_detection(self, residuals_normalized):
        residuals_standardized = residuals_normalized
        inliers = np.array((np.abs(residuals_standardized) < self.T_threshold))
        return ~(inliers)
    
    def fit(self, X, y):
        s = self.signal_preprocessing(y)
        self.regressor.fit(X, s)
        return self
    def predict(self, X):
        return self.signal_unpreprocessing(self.regressor.predict(X))  
    
def shore_matrix(bvals, bvecs, radial_order, zeta, tau, affine = None):
    gtab = gradient_table(bvals, bvecs)
    Phi = dipy.reconst.shore.shore_matrix(radial_order, zeta, gtab, tau)
    return Phi   

def IRL1SHORE():
    parameters = config['IRL1SHORE']
    
    radial_order = parameters.getint('radial_order')
    zeta         = parameters.getint('zeta')
    tau          = parameters.getfloat('tau')
    
    X = shore_matrix(bvals, bvecs,radial_order, zeta, tau)
    
    data_reshaped = data.reshape((-1,data.shape[3]))
    
    if mask is not None:
        mask_reshaped = mask.flatten()
        mask_reshaped = mask_reshaped ==1
        data_masked   = data_reshaped[mask_reshaped]
    else:
        data_masked = data_reshaped
        
    result_residuals   = np.zeros(data_masked.shape, dtype=np.float64)
    result_outliers    = np.zeros(data_masked.shape, dtype=np.bool)
    result_inliers     = np.zeros(data_masked.shape, dtype=np.bool)
    result_predictions = np.zeros(data_masked.shape, dtype=np.float64)
    
    for i in range(data_masked.shape[0]):
        y = data_masked[i]
        model = IRLSHORE(parameters.getfloat('lambda'), 
                         parameters.getint('max_iter'), 
                         parameters['residual_calculation'],
                         parameters.getboolean('verbose'),
                         parameters.getfloat('t_threshold'),
                         0,
                         parameters.getboolean('one_sided_reweighting'),
                         parameters.getboolean('one_sided_scoring'), 
                         parameters.getfloat('convergence_threshold'),
                         radial_order,
                         zeta,
                         tau,
                         S0 = np.mean(y[bvals ==0]))
        
        residuals = model.outlier_scoring(X, y)
        outliers = model.outlier_detection(residuals)
        inliers = (~outliers)
        model.fit(X[inliers], y[inliers])
        predictions = model.predict(X)
    
        result_residuals[i] = residuals
        result_outliers[i] = outliers
        result_inliers[i] = inliers
        result_predictions[i] = predictions
    
    if mask is not None:
        result_residuals_unmasked = unmask(result_residuals, mask_reshaped)
        result_outliers_unmasked = unmask(result_outliers, mask_reshaped)
        result_inliers_unmasked = unmask(result_inliers, mask_reshaped)
        result_predictions_unmasked = unmask(result_predictions, mask_reshaped)
    else:
        result_residuals_unmasked = result_residuals
        result_outliers_unmasked  = result_outliers
        result_inliers_unmasked  = result_inliers
        result_predictions_unmasked  = result_predictions
        
        
        
    predictions = pd.DataFrame(result_predictions_unmasked)
    inliers     = pd.DataFrame(result_inliers_unmasked)
    
    if parameters['imputation'] == 'replace_outliers':
        correct_data = pd.DataFrame(data_reshaped)
        correct_data = correct_data[inliers]
        correct_data[correct_data.isnull()] = predictions
        pd.to_pickle(correct_data, 'output_IRL1SHORE_' + data_file.rstrip('.nii') + '_corrected_data.pkl')
       
    elif parameters['imputation'] == 'replace_all':
        pd.to_pickle(predictions, 'output_IRL1SHORE_' + data_file.rstrip('.nii') + '_predictions.pkl') 
     
        
    if parameters.getboolean('score_file'):
        residuals = pd.DataFrame(result_residuals_unmasked)
        pd.to_pickle(residuals, 'output_IRL1SHORE_' + data_file.rstrip('.nii') + '_residuals_score.pkl')
        


    
def relSHORE():
    parameters = config['relSHORE']
   
    radial_order = parameters.getint('radial_order')
    zeta         = parameters.getint('zeta')
    tau          = parameters.getfloat('tau')
    
    X = shore_matrix(bvals, bvecs,radial_order, zeta, tau)
    
    data_reshaped = data.reshape((-1,data.shape[3]))
    
    if mask is not None:
        mask_reshaped = mask.flatten()
        mask_reshaped = mask_reshaped ==1
        data_masked   = data_reshaped[mask_reshaped]
    else:
        data_masked = data_reshaped
        
    result_residuals   = np.zeros(data_masked.shape, dtype=np.float64)
    result_outliers    = np.zeros(data_masked.shape, dtype=np.bool)
    result_inliers     = np.zeros(data_masked.shape, dtype=np.bool)
    result_predictions = np.zeros(data_masked.shape, dtype=np.float64)
    
    for i in range(data_masked.shape[0]):
        y = data_masked[i]
        model = IRLSHORE(parameters.getfloat('lambda'), 
                         0, 
                         'relative',
                         parameters.getboolean('verbose'),
                         parameters.getfloat('t_threshold'),
                         parameters.getfloat('power_relative'),
                         False,
                         parameters.getboolean('one_sided_scoring'), 
                         0,
                         radial_order,
                         zeta,
                         tau,
                         S0 = np.mean(y[bvals ==0]))
        
        residuals = model.outlier_scoring(X, y)
        outliers = model.outlier_detection(residuals)
        inliers = (~outliers)
        model.fit(X[inliers], y[inliers])
        predictions = model.predict(X)
    
        result_residuals[i] = residuals
        result_outliers[i] = outliers
        result_inliers[i] = inliers
        result_predictions[i] = predictions
    
    if mask is not None:
        result_residuals_unmasked = unmask(result_residuals, mask_reshaped)
        result_outliers_unmasked = unmask(result_outliers, mask_reshaped)
        result_inliers_unmasked = unmask(result_inliers, mask_reshaped)
        result_predictions_unmasked = unmask(result_predictions, mask_reshaped)
    else:
        result_residuals_unmasked = result_residuals
        result_outliers_unmasked  = result_outliers
        result_inliers_unmasked  = result_inliers
        result_predictions_unmasked  = result_predictions
        
        
        
    predictions = pd.DataFrame(result_predictions_unmasked)
    inliers     = pd.DataFrame(result_inliers_unmasked)
    
    if parameters['imputation'] == 'replace_outliers':
        correct_data = pd.DataFrame(data_reshaped)
        correct_data = correct_data[inliers]
        correct_data[correct_data.isnull()] = predictions
        pd.to_pickle(correct_data, 'output_relSHORE_' + data_file.rstrip('.nii') + '_corrected_data.pkl')
       
    elif parameters['imputation'] == 'replace_all':
        pd.to_pickle(predictions, 'output_relSHORE_' + data_file.rstrip('.nii') + '_predictions.pkl') 
     
        
    if parameters.getboolean('score_file'):
        residuals = pd.DataFrame(result_residuals_unmasked)
        pd.to_pickle(residuals, 'output_relSHORE_' + data_file.rstrip('.nii') + '_residuals_score.pkl')
        
        
        
if model == 'REKINDLE':
    rekindle()
elif model == 'IRL1SHORE':
    IRL1SHORE()
elif model == 'relSHORE':
    relSHORE()
   
    

   