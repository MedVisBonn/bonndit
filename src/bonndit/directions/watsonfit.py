import numpy as np
from bonndit.utils.watson_utils import fodf_to_watson_sh, watson_sh_to_fodf, lowrankapprox_from_fodf, nrrd_peaks_split, create_missing_folders, save_as_fodf_nrrd, EPS_L, CONF_90, O_LIMIT
from dipy.reconst.forecast import cart2sphere
from dipy.core.geometry import sphere2cart
import timeit
import logging
import sys
import nrrd
import multiprocessing
from itertools import repeat
import pyshtools as pysh
import bonndit.utilc.watsonfitwrapper as watsonfit
import bonndit.utils.watson_fodf_generation as wfg

class WatsonFit(object):
    def __init__(self, shorder, init='lowrank', kappa_range=(39.9,40), rank=3, wmmin=0.3, no_spread=False, verbose=False):
        """Model fODFs with multiple Watson distributions

        Args:
            shorder: spherical harmonics order, same as the tensor order
            init: How the fitting should be initialized, defaults to 'lowrank' for the lowrank fit by Schultz and Seidel, 2008. Alternative 'rand' for random init or 'given' for given values
            kappa_range: Range of initial kappa values to randomly sample from, defaults to (39.9,40).
            rank: number of distributions to fit, allowed values are 2 and 3. Defaults to 3.
            wmmin: minimal White Matter density, defaults to 0.3.
            no_spread: if set to True only the rank1 tensors are fitted without spread.
            verbose: logging steps if set to True
        """
        self.shorder = shorder
        self.init = init
        self.kappa_range = kappa_range
        self.rank = rank
        self.wmmin = wmmin
        self.no_spread = no_spread
        self.verbose = verbose

        # enable logging if verbose
        if self.verbose:
            logging.basicConfig(format='%(asctime)s: %(levelname)-8s %(message)s',
                     level=logging.INFO, stream=sys.stdout)

    def fit(self, fodf, fodf_header, wm_data=None, initvals=None, init_kappa_values=None, outlier_handling=True):
        """Fit Watson distributions to given fODFs

        Args:
            fodf: fODFs to be fitted
            fodf_header: Header of the fODFs to be fitted
            wm_data: white matter density information e.g. from wmvolume.nrrd, if None, all voxels are used
            initvals: direction and volume fraction values for the initialization, e.g. from lowrank nrrd, alternative to init variable. Defaults to None.
            init_kappa_values: kappa values for the initialization. Defaults to None.
            outlier_handling: if set to False the Outlier detection and handling is turned off.
        """
        self.fodf_header = fodf_header
        # convert tensor rep. to sh rep.
        if self.verbose:
            if self.no_spread:
                logging.info("No Spread, fitting rank-one tensors")
            logging.info("Generating SH coefficients for fODFs")
        fodf_sh = fodf_to_watson_sh(fodf, self.shorder)
        
        # initial values
        if self.verbose:
            logging.info("Generating initial values")
        if self.init == 'lowrank':
            init_peak_values, init_peak_dirs = lowrankapprox_from_fodf(fodf, self.rank, self.verbose)
        elif self.init == 'given':
            if initvals is None:
                raise TypeError('No initial values specified with init set to \'given\'.')
            init_peak_values, init_peak_dirs = nrrd_peaks_split(initvals)
        
        self.watson_model = WatsonInitModel(self.kappa_range, self.shorder, self.rank, self.wmmin, self.no_spread)
        fitting_params = self.watson_model.fitting_params(fodf_sh, wm_data, init_peak_dirs, init_peak_values, self.kappa_range)

        if self.verbose:
            logging.info("Watson fitting")
        start = timeit.default_timer()
        watsonfit.mw_openmp_mult_p(*fitting_params)
        
        print("\nRuntime:", timeit.default_timer() - start, "seconds")
        print(self.watson_model.loss.mean())

        # outlier handling
        if outlier_handling:
            if self.verbose:
                logging.info("Outlier handling")
            outlier_params = self.watson_model.outlier_params(O_LIMIT[self.shorder])
            watsonfit.mw_openmp_mult_p(*outlier_params)
            print("\n")

        # post processing
        self.watson_model.process_results()

        # save to results
        self.result_model = WatsonResultModel(self.watson_model.estimation, 
                                            self.watson_model.estimation_loss, 
                                            fodf_sh, 
                                            fodf_header,
                                            init_peak_dirs, 
                                            init_peak_values,
                                            self.verbose)

        return self.result_model

class WatsonInitModel(object):
    def __init__(self, kappa_range, shorder, rank, wmmin, no_spread = False):
        self.kappa_range = kappa_range
        self.shorder = shorder
        self.rank = rank
        self.wmmin = wmmin
        self.no_spread = no_spread

    def fitting_params(self, fodf_sh, wm_data, init_peak_dirs, init_peak_values, init_kappa_values):
        self.init_peak_dirs = init_peak_dirs
        self.init_peak_values = init_peak_values
        self.init_kappa_values = init_kappa_values
        self.fodf_sh = fodf_sh

        # prepare values
        filtered_fodf_sh = fodf_sh.copy()
        if wm_data is not None:
            white_matter_wmd = wm_data > self.wmmin
            filtered_fodf_sh[~white_matter_wmd] = 0

        self.cimax = filtered_fodf_sh.shape[3]

        # signals to estimate, reshaped to form a list
        self.signals = filtered_fodf_sh.reshape(-1, self.cimax)

        # filter for non-zero
        self.nonzero_mask = np.where(np.sum(np.abs(self.signals), axis = 1) > EPS_L)
        self.fitting_signals = self.signals[self.nonzero_mask]

        # compute initial params from values
        t = init_peak_dirs[:,:,:,:self.rank].reshape(-1, 3)
        euler_angles = np.array(cart2sphere(-t[:,0],t[:,1],-t[:,2])).T[:,1:]
        #w_values = np.random.uniform(0.1,1.0,(len(euler_angles),1))
        w_values = init_peak_values[:,:,:,:self.rank].reshape(-1,1)
        k_values = np.random.uniform(np.log(self.kappa_range[0]),np.log(self.kappa_range[1]),(len(euler_angles),1))

        # merge params
        self.b_fitting_init = np.hstack([w_values,k_values,euler_angles]).reshape(len(self.signals),-1)[self.nonzero_mask]

        # set all fitting params
        self.fitting_init = self.b_fitting_init.copy()
        self.amount = self.fitting_init.shape[0]
        self.loss = np.zeros(self.amount)
        angles_v = np.zeros((self.amount, 3))
        dipy_v = np.zeros((self.amount, self.cimax))
        pysh_v = np.zeros((self.amount, 2, self.shorder+1, self.shorder+1))
        rot_pysh_v = np.zeros_like(pysh_v)
        est_signal = np.zeros((self.amount, self.cimax))

        return (self.fitting_init, self.fitting_signals, est_signal, dipy_v, 
        pysh_v, rot_pysh_v, angles_v, self.loss, self.amount, self.shorder, self.rank, 1 if self.no_spread else 0)

    def outlier_params(self, cutoff):
        self.outlier_mask = self.loss > cutoff
        self.outlier_fitting_init = self.b_fitting_init[self.outlier_mask].copy()

        w_values = np.random.uniform(0.1,1.0,(len(self.outlier_fitting_init)*self.rank,1))
        k_values = np.random.uniform(np.log(self.kappa_range[0]),np.log(self.kappa_range[1]),(len(self.outlier_fitting_init)*self.rank,1))
        self.outlier_fitting_init.reshape(-1,4)[:,:2] = np.hstack([w_values, k_values])

        outlier_fitting_signals = self.fitting_signals[self.outlier_mask].copy()

        # set all fitting params
        o_amount = self.outlier_fitting_init.shape[0]
        self.o_loss = np.zeros(o_amount)
        angles_v = np.zeros((o_amount, 3))
        dipy_v = np.zeros((o_amount, self.cimax))
        pysh_v = np.zeros((o_amount, 2, self.shorder+1, self.shorder+1))
        rot_pysh_v = np.zeros_like(pysh_v)
        est_signal = np.zeros((o_amount, self.cimax))

        return (self.outlier_fitting_init, outlier_fitting_signals, est_signal, dipy_v, 
        pysh_v, rot_pysh_v, angles_v, self.o_loss, o_amount, self.shorder, self.rank, 1 if self.no_spread else 0)

    def process_results(self):
        # outlier handling
        if self.outlier_fitting_init is not None:
            tmp_loss = self.loss.copy()
            tmp_loss[self.outlier_mask] = self.o_loss
            tmp_init_values = self.fitting_init.copy()
            tmp_init_values[self.outlier_mask] = self.outlier_fitting_init
            smaller_loss_mask = tmp_loss < self.loss
            self.loss[smaller_loss_mask] = tmp_loss[smaller_loss_mask]
            self.fitting_init[smaller_loss_mask] = tmp_init_values[smaller_loss_mask]

        # watson params
        estimation = np.zeros((len(self.signals), self.rank*4))
        estimation[self.nonzero_mask] = self.fitting_init
        estimation = estimation.reshape(-1, 4)
        weight = np.abs(estimation[:,0])
        weight = np.nan_to_num(weight, neginf=0, posinf=0)
        kappa = np.exp(estimation[:,1])
        kappa = np.nan_to_num(kappa, neginf=0, posinf=0)

        # re-combine results
        abs_estimation = np.hstack([weight[:,np.newaxis], kappa[:,np.newaxis], estimation[:,2:4]]).reshape(-1,self.rank,4)

        # order by volume fraction
        sortvals = np.argsort(abs_estimation[:,:,0], axis=1)
        tiled_sortvals = np.moveaxis(np.tile(sortvals[:,::-1],4).reshape(-1,4,self.rank), [-1, -2], [-2, -1])
        abs_estimation_sorted = np.take_along_axis(abs_estimation, tiled_sortvals, axis=1)

        # reshape results to align with original shape
        self.estimation = abs_estimation_sorted.reshape(*self.init_peak_dirs.shape[:4], 4)

        # loss
        estimation_loss = np.zeros(len(self.signals))
        estimation_loss[self.nonzero_mask] = self.loss
        # reshape loss to align with original shape
        self.estimation_loss = estimation_loss.reshape(self.init_peak_dirs.shape[:3])


class WatsonResultModel(object):
    def __init__(self, estimation, estimation_loss, fodf_sh, fodf_header, init_peak_dirs, init_peak_values, verbose=False):
        self.estimation = estimation
        self.estimation_loss = estimation_loss
        self.fodf_sh = fodf_sh
        self.fodf_header = fodf_header
        self.init_peak_dirs = init_peak_dirs
        self.init_peak_values = init_peak_values
        self.rank = estimation.shape[-2]
        self.verbose = verbose
        self.shorder = int(0.5*(np.sqrt(1 + 8*(fodf_sh.shape[-1]))-3))

        reshaped_results = self.estimation.reshape(-1, 4)
        self._cart_angles = np.array(sphere2cart(1, reshaped_results[:,2], reshaped_results[:,3])).T

        self._reshaped_weight = reshaped_results[:,0]
        self._reshaped_kappa = reshaped_results[:,1]

        # set directions to zero where weight is zero
        zero_weight_mask = self._reshaped_weight == 0
        self._cart_angles[zero_weight_mask,:] = 0

        # directions mirrored on the Z axis to align with vvi
        self._cart_angles[:,0] = -self._cart_angles[:,0]
        self._cart_angles[:,2] = -self._cart_angles[:,2]

        self.kappas = self._reshaped_kappa.reshape(*self.estimation.shape[:3],self.rank)
        self.weights = self._reshaped_weight.reshape(*self.estimation.shape[:3],self.rank)
        self.directions = self._cart_angles.reshape(*self.estimation.shape[:3],self.rank,3)

        # modify fodf header if too many rows
        if self.fodf_header['space directions'].shape[0] > 3:
            self.fodf_header['space directions'] = self.fodf_header['space directions'][1:4,:3]

        # enable logging if verbose
        if verbose:
            logging.basicConfig(format='%(asctime)s: %(levelname)-8s %(message)s',
                     level=logging.INFO, stream=sys.stdout)

    def export_for_watson_tracking(self, filename):
        # combine weights and angles
        cart_results = np.hstack([self._reshaped_kappa[:,np.newaxis], self._reshaped_weight[:,np.newaxis], self._cart_angles]).reshape(-1,self.rank,5)

        # reshape
        cart_results = cart_results.reshape(*self.estimation.shape[:3],self.rank,5)

        # reorder for nrrd
        cart_results = np.moveaxis(cart_results, [-1, -2], [0, 1])
        cart_results = cart_results[:,:self.rank,...]

        # save to nrrd output file
        create_missing_folders(filename)
        newmeta = {k: self.fodf_header[k] for k in ['space', 'space origin']}
        newmeta['kinds'] = ['list', 'list', 'space', 'space', 'space']
        newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], self.fodf_header['space directions']))
        nrrd.write(filename, np.float32(cart_results), newmeta)

    def export_for_peak_tracking(self, filename):
        # combine weights and angles
        cart_results = np.hstack([self._reshaped_weight[:,np.newaxis], self._cart_angles]).reshape(-1,self.rank,4)

        # reshape
        cart_results = cart_results.reshape(*self.estimation.shape[:3],self.rank,4)

        # reorder for nrrd
        cart_results = np.moveaxis(cart_results, [-1, -2], [0, 1])

        # save to nrrd output file
        create_missing_folders(filename)
        newmeta = {k: self.fodf_header[k] for k in ['space', 'space origin']}
        newmeta['kinds'] = ['list', 'list', 'space', 'space', 'space']
        newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], self.fodf_header['space directions']))
        nrrd.write(filename, np.float32(cart_results), newmeta)

    def export_for_vvi_with_cones(self, filename):
        confidence = np.ones_like(self._reshaped_weight)

        # scale cones by volume fraction
        scaled_angles = self._cart_angles * self._reshaped_weight[:,None]

        # compute opening angle (using precomputed 90p interval)
        kappa_with_limit = np.clip(self._reshaped_kappa.copy(),0,190)
        kappa_angle = np.radians(CONF_90(kappa_with_limit))
        
        # combine values
        cart_results = np.hstack([  confidence[:,np.newaxis], confidence[:,np.newaxis], 
                                    kappa_angle[:,np.newaxis], scaled_angles]).reshape(-1,self.rank,6)

        # reshape
        cart_results = cart_results.reshape(*self.estimation.shape[:3],self.rank,6)

        # reorder for nrrd
        cart_results = np.moveaxis(cart_results, [-1, -2], [0, 1])

        # save to nrrd output file
        newmeta = {k: self.fodf_header[k] for k in ['space', 'space origin']}
        newmeta['kinds'] = ['list', 'space', 'space', 'space']
        newmeta['space directions'] = np.vstack(([np.nan, np.nan, np.nan], self.fodf_header['space directions']))

        # save to nrrd output files for all directions
        create_missing_folders(filename)
        for i in range(self.rank):
            if self.verbose:
                logging.info("Saving " + filename.removesuffix('.nrrd') + str(i+1) + ".nrrd")
            cart_results_one = cart_results[:,i,...]
            nrrd.write(filename.removesuffix('.nrrd') + str(i+1) + ".nrrd", np.float32(cart_results_one), newmeta)

    def export_as_fodf_signal(self, filename, peak_number = None):
        only_one_peak = False
        if peak_number is not None:
            only_one_peak = True

        # rotation matrix
        dj = pysh.rotate.djpi2(self.shorder)

        reshaped_results = self.estimation.reshape(-1, 4*self.rank)

        # only compute for non zero signals
        nonzero_results_mask = np.abs(reshaped_results[:,0]) > 0

        generated_signals = np.zeros((len(reshaped_results), self.fodf_sh.shape[-1]))

        zipped_params = zip(reshaped_results[nonzero_results_mask],
                            repeat(self.rank),
                            repeat(dj),
                            repeat(True),
                            repeat(only_one_peak),
                            repeat(peak_number),
                            repeat(False),
                            repeat(0),
                            repeat(self.shorder))

        # generate signals as sh
        results = None
        a_pool = multiprocessing.Pool()
        results = a_pool.starmap(wfg.watson_fodf_signals_generator, zipped_params, chunksize=int(len(reshaped_results[nonzero_results_mask]) / a_pool._processes / 5))

        generated_signals[nonzero_results_mask] = np.array(results)

        generated_signals = generated_signals.reshape(*self.estimation.shape[:3], self.fodf_sh.shape[-1])

        # convert sh to tensors
        fodf_data = watson_sh_to_fodf(generated_signals, self.shorder)

        save_as_fodf_nrrd(filename, fodf_data, self.fodf_header)       
        
    def backup_to_file(self, filename):
        """helper method to backup results

        Args:
            filename: filename for backup
        """
        create_missing_folders(filename)
        np.savez_compressed(filename,
                    watson=self.estimation, 
                    loss=self.estimation_loss, 
                    shm_coeff=self.fodf_sh, 
                    fodf_header=self.fodf_header,
                    peak_dirs=self.init_peak_dirs, 
                    peak_values=self.init_peak_values, 
                    x_r=(0,self.init_peak_dirs.shape[0]),
                    y_r=(0,self.init_peak_dirs.shape[1]),
                    z_r=(0,self.init_peak_dirs.shape[2]))

    @staticmethod
    def load_from_file(filename, verbose=False):
        """helper method load results from file

        Args:
            filename: filename for backup
        """
        loaded = np.load(filename, allow_pickle=True)

        result_model = WatsonResultModel(loaded['watson'], loaded['loss'], loaded['shm_coeff'], loaded['fodf_header'].item(), loaded['peak_dirs'], loaded['peak_values'], verbose)
        return result_model