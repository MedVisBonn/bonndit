from __future__ import division
import numpy as np
import dipy
import nibabel as nib
from dipy.reconst.shore import shore_matrix
from dipy.core.gradients import gradient_table
import sklearn
from sklearn import linear_model
import configparser
from collections import OrderedDict


def LLSfit(X, y, regularization_constant):
    """
    Fits Linear Least Squares
    :param X: Design matrix
    :param y: Preprocessed data vector
    :param regularization_constant: Prevents matrix inversion from failure when inverting degenerate matrices
    :return: Parameters vector beta by linear least squares fit
    """
    return np.linalg.inv(X.transpose() * X + np.matrix(np.eye(X.shape[1])) * regularization_constant) * \
        X.transpose() * y


def WLLSfit(X, y, W, regularization_constant):
    """
    Fits Weighted Linear Least Squares given weights matrix
    :param X: Design matrix
    :param y: Preprocessed data vector
    :param W: Weights matrix
    :param regularization_constant: Prevents matrix inversion from failure when inverting degenerate matrices
    :return: Parameters vector beta by weighted linear least squares fit
    """
    return np.linalg.inv(X.transpose() * W * X + np.matrix(np.eye(X.shape[1])) * regularization_constant) * \
        X.transpose() * W * y


# Median absolute deviation given 1d array
MAD = lambda residuals: np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)


class REKINDLE(object):
    def __init__(self,
                 regularization_constant,
                 kappa,
                 c,
                 IRLS_maxiter,
                 REKINDLE_maxiter,
                 verbose):
        self.regularization_constant = regularization_constant
        self.kappa = kappa
        self.c = c
        self.IRLS_maxiter = IRLS_maxiter
        self.REKINDLE_maxiter = REKINDLE_maxiter
        self.verbose = verbose

    def conv_check(self, beta1, beta2):
        # Calculates convergence criterion (<0 is good)
        return (np.abs(beta1-beta2) - self.c * np.max(np.concatenate((np.abs(beta1), np.abs(beta2)), axis=1), axis=1))\
            .max()

    def IRLS__(self, X, y, beta_init, weighting='geman-mcclure'):
        # Iteratively Reweighted Least Squares (linear) weighting parameter according to weighting used
        beta_old = beta_init

        counter = 0
        tag = True

        while tag:
            if weighting == 'geman-mcclure':
                residuals = np.array(y - X * beta_old)
                residuals_sigma = 1.4826 * MAD(residuals)
                residuals_standardized = residuals / residuals_sigma
                weights = 1.0 / (residuals_standardized**2 + 1)**2
            elif weighting == 'exponential':
                weights = np.exp(2 * np.array(X*beta_old))

            W = np.matrix(np.eye(X.shape[0]) * weights)

            beta_new = WLLSfit(X, y, W, self.regularization_constant)

        # convergence_check = (np.abs(beta_new-beta_old) - c* np.max(np.concatenate((np.abs(beta_new),
        # np.abs(beta_old)), axis= 1), axis=1)).max()
            convergence_check = self.conv_check(beta_new, beta_old)

            if self.verbose:
                print('Iteration {:03}. Convergence criterion: {:.2e}'.format(counter+1, convergence_check))

            if convergence_check > 0:
                beta_old = beta_new.copy()
                counter += 1
            else:
                tag = False
                if self.verbose:
                    print('Exiting due to convergence.')

            if counter > self.IRLS_maxiter:
                tag = False
                if self.verbose:
                    print('Exiting due to maximum number of iterations reached. ')
        return beta_new

    # Signal preprocessing in the logarithmized space
    @staticmethod
    def signal_preprocessing(y):
        # Converting the data input vector to the logarithmized space
        assert len(y.shape) == 1
        return np.log(y)

    @staticmethod
    def signal_unpreprocessing(y):
        # Converting data input vector from the logarithmized space back to original values
        assert len(y.shape) == 1
        return np.exp(y)

    def outlier_scoring(self, X, y):
        s = self.signal_preprocessing(y)
        # Removing impossible values from further fit
        inliers = ~np.isnan(s)
        s = np.matrix(s).transpose()
        # inliers[y < 1e-10] = False
        tag = True
        counter = 0
        beta_old = LLSfit(X[inliers], s[inliers], self.regularization_constant)

        while tag:
            beta_robust = self.IRLS__(X[inliers], s[inliers], beta_old)
            s_star = s / np.exp(-X * beta_robust)
            X_star = X / np.exp(-X * beta_robust)

            beta_rescaled = LLSfit(X_star[inliers], s_star[inliers], self.regularization_constant)
            beta_new = self.IRLS__(X_star[inliers], s_star[inliers], beta_rescaled)

            convergence_check = self.conv_check(beta_new, beta_old)

            if self.verbose:
                    print('Iteration {:03}. Convergence criterion: {:.2e}'.format(counter+1, convergence_check))
            if convergence_check > 0:
                beta_old = beta_new.copy()
                counter += 1
            else:
                if self.verbose:
                    print('Exiting due to convergence.')
                tag = False
            if counter > self.REKINDLE_maxiter:
                tag = False
                if self.verbose:
                    print('Exiting due to maximum number of iterations reached. ')

        residuals_star = s_star - X_star * beta_new
        residuals_sigma = 1.4826 * MAD(residuals_star[inliers])
        residuals_normalized = residuals_star.flatten()/residuals_sigma
        return residuals_normalized

    def outlier_detection(self, residuals_normalized):
        # Detecting outliers given a specified threshold kappa
        inliers = (np.array(np.abs(residuals_normalized) < self.kappa).ravel())
        outliers = ~inliers
        return tuple((inliers, outliers))

    def fit(self, X, y):
        # Final fit without outliers
        s = self.signal_preprocessing(y)
        s = np.matrix(s).transpose()
        self.beta = self.IRLS__(X, s, LLSfit(X, s, self.regularization_constant), weighting='exponential')
        return self.beta

    def predict(self, X):
        # Predicting signals using parameters vector beta and design matrix X
        # beta = beta.reshape((22,1))
        predict_matrix = X * self.beta
        return self.signal_unpreprocessing(np.array(predict_matrix).flatten())


def design_matrix(bvals, bvecs, affine):
    """
    Creates a design matrix X
    :param bvals: b-values
    :param bvecs: b-vectors
    :param affine: affine values
    :return: design matrix X
    """

    # re-scale b values to more natural units
    bvals /= 1000.0

    # flip the signs of bvec coordinates as needed
    for i in range(3):
        if affine[i, i] < 0:
            bvecs[:, i] = -bvecs[:, i]

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


class IRLSHORE(object):
    def __init__(self,
                 Lambda,
                 max_iter,
                 residual_calculation,
                 T_threshold,
                 power_relative,
                 one_sided_reweighting,
                 one_sided_scoring,
                 convergence_threshold,
                 radial_order,
                 zeta,
                 tau,
                 verbose,
                 S0=0):
        self.max_iter = max_iter
        self.Lambda = Lambda
        self.S0 = S0
        self.T_threshold = T_threshold
        self.one_sided_reweighting = one_sided_reweighting
        self.one_sided_scoring = one_sided_scoring
        self.convergence_threshold = convergence_threshold
        self.radial_order = radial_order
        self.zeta = zeta
        self.tau = tau
        self.verbose = verbose
        if self.Lambda == 0:
            self.regressor = linear_model.LinearRegression()
        else:
            self.regressor = linear_model.Lasso(alpha=self.Lambda)

        # Specifying a method to calculate the residues, 'relative' for relSHORE model
        if residual_calculation == 'REKINDLE':
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
        # Normalizing signal by S0 value
        return y / self.S0

    def signal_unpreprocessing(self, y):
        return y * self.S0

    def compute_residuals_REKINDLE(self, s_true, s_pred, inliers, one_sided=False):
        residuals = s_true - s_pred
        residuals_sigma = 1.4826 * MAD(residuals[inliers])
        residuals_standardized = residuals / residuals_sigma
        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized

    def compute_residuals_relative(self, s_true, s_pred, inliers, one_sided=False):
        residuals = (s_true - s_pred)/(np.clip(s_pred, 1e-3, np.inf) ** self.power_relative)
        residuals_sigma = 1.4826 * MAD(residuals[inliers])
        residuals_standardized = residuals / residuals_sigma
        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized

    def compute_residuals_REKINDLE_log(self, s_true, s_pred, inliers, one_sided=False):
        S_true = self.signal_unpreprocessing(s_true)
        S_log = np.log(S_true)
        s_pred[s_pred < 1e-3] = 1e-3

        S_pred_log = np.log(self.signal_unpreprocessing(s_pred))

        residuals = S_log - S_pred_log
        residuals_sigma = 1.4826 * MAD(residuals[inliers])
        residuals_standardized = residuals / residuals_sigma

        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized

    def compute_residuals_SHORE(self, s_true, s_pred, inliers, one_sided=False):
        residuals = s_true - s_pred
        residuals_sigma = 1.4826 * MAD(residuals[inliers])
        s0_pred = self.regressor.predict(
            shore_matrix(np.array(0, ndmin = 1), np.array([0,0,0], ndmin=2),self.radial_order, self.zeta, self.tau))

        residuals_standardized = (s0_pred * s_true - s_pred)/ (residuals_sigma * ((s_true**2 + 1)**0.5))
        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized

    def compute_residuals_SHORE_log(self, s_true, s_pred, inliers, one_sided=False):
        S_true = self.signal_unpreprocessing(s_true)
        S_log = np.log(S_true)
        s_pred[s_pred < 1e-3] = 1e-3
        S_hat_log = np.log(self.signal_unpreprocessing(s_pred))

        residuals = S_log - S_hat_log
        residuals_sigma = 1.4826 * MAD(residuals[inliers])
        residuals_standardized = (S_log - S_hat_log)/ (residuals_sigma * ((S_log**2 + 1)**0.5))

        if one_sided:
            residuals_standardized = np.clip(residuals_standardized, -np.inf, 0)
        return residuals_standardized

    def outlier_scoring(self, X, y):
        """
        Scores outliers by fitting the data using linear regression or LASSO iteratively by reweighting
        :param X: Design matrix
        :param y: Input data vector
        :return: residuals_standardized
        """
        inliers = np.ones((y.shape[0]), dtype=np.bool)
        inliers[y < 1e-10] = False
        if ~np.any(inliers):
            return -np.inf * np.ones(y.shape)

        # Normalizing signal
        s = self.signal_preprocessing(y)
        self.regressor.fit(X[inliers], s[inliers])
        # beta_old = self.regressor.coef_.copy()
        omega_old = np.zeros(y.shape)
        iter_count = 0
        while True:
            iter_count += 1
            if iter_count >= self.max_iter:
                # beta_new = self.regressor.coef_.copy()
                if self.verbose:
                    print('Exiting due to maximum number of iterations reached')
                break
            s_hat = self.regressor.predict(X)
            residuals_standardized = self.compute_residuals(s, s_hat, inliers, one_sided=self.one_sided_reweighting)
            omega = 1.0/(residuals_standardized**2 + 1)
            assert ~np.isnan(omega).any()
            self.regressor.fit(np.dot(np.diag(omega[inliers]), X[inliers]), omega[inliers] * s[inliers])

            # beta_new = self.regressor.coef_.copy()
            # convergence_check = np.linalg.norm(beta_new-beta_old, ord = 2)       # Original proposition
            convergence_check = np.linalg.norm(omega - omega_old, ord=2)

            if convergence_check < self.convergence_threshold:
                if self.verbose:
                    print('Exiting due to convergence')
                break
            omega_old = omega.copy()
            # beta_old = beta_new

        s_hat = self.regressor.predict(X)
        residuals_standardized = self.compute_residuals(s, s_hat, inliers, one_sided=self.one_sided_scoring)
        return residuals_standardized

    def outlier_detection(self, residuals_normalized):
        residuals_standardized = residuals_normalized
        inliers = np.array((np.abs(residuals_standardized) < self.T_threshold))
        return ~inliers

    def fit(self, X, y):
        if y.shape[0] == 0:
            return np.nan * np.ones(X.shape[1])
        s = self.signal_preprocessing(y)
        self.regressor.fit(X, s)
        return self.regressor.coef_.copy()

    def predict(self, X, y):
        if y.shape[0] == 0:
            return np.nan * np.ones(X.shape[0])
        return self.signal_unpreprocessing(self.regressor.predict(X))


def shore_matrix(bvals, bvecs, radial_order, zeta, tau):
    """
    Creates a SHORE matrix
    :param bvals: b-values
    :param bvecs: b-vectors
    :param radial_order:
    :param zeta:
    :param tau:
    :return: SHORE matrix Phi
    """
    gtab = gradient_table(bvals, bvecs)
    Phi = dipy.reconst.shore.shore_matrix(radial_order, zeta, gtab, tau)
    return Phi


def create_parameters_file(dir_path):
    # Creates a parameters file if none exists in the input folder or if none is given by the user
    config = configparser.ConfigParser()
    config['REKINDLE'] = OrderedDict([('regularization_constant', '0.00001'),
                                      ('kappa', '2.5'),
                                      ('c', '0.01'),
                                      ('IRLS_maxiter', '20'),
                                      ('REKINDLE_maxiter', '20'),
                                      ('score_file', 'False'),
                                      ('log_file', 'False'),
                                      ('beta_file', 'False'),
                                      ('pickle_files', 'False'),
                                      ('imputation', 'replace_outliers')])

    config['IRL1SHORE'] = OrderedDict([('Lambda', '0.0000001'),
                                      ('max_iter', '20'),
                                      ('residual_calculation', 'SHORE'),
                                      ('radial_order', '4'),
                                      ('zeta', '400'),
                                      ('tau', '0.0344'),
                                      ('T_threshold', '2.0'),
                                      ('one_sided_reweighting', 'False'),
                                      ('one_sided_scoring', 'True'),
                                      ('convergence_threshold', '0.00001'),
                                      ('score_file', 'False'),
                                      ('log_file', 'False'),
                                      ('beta_file', 'False'),
                                      ('pickle_files', 'False'),
                                      ('imputation', 'replace_outliers')])

    config['relSHORE'] = OrderedDict([('Lambda', '0.0000001'),
                                      ('radial_order', '4'),
                                      ('zeta', '400'),
                                      ('tau', '0.0344'),
                                      ('T_threshold', '2.0'),
                                      ('power_relative', '0.5'),
                                      ('one_sided_scoring', 'True'),
                                      ('score_file', 'False'),
                                      ('log_file', 'False'),
                                      ('beta_file', 'False'),
                                      ('pickle_files', 'False'),
                                      ('imputation', 'replace_outliers')])

    with open(dir_path + '/' + 'parameters.ini', 'w') as configfile:
        config.write(configfile)

    print('A parameters file with default parameters has been created in the data folder')
