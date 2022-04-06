#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import sys
import nrrd
sys.path.append('.')
from .alignedDirection cimport  Gaussian, Laplacian, ScalarOld, ScalarNew, Probabilities, Deterministic,Deterministic2
from .ItoW cimport Trafo
from .stopping cimport Validator
from .integration cimport  Euler, Integration
from .interpolation cimport  FACT, Trilinear, Interpolation, UKFFodf, UKFMultiTensor
from bonndit.utilc.cython_helpers cimport mult_with_scalar, sum_c, sum_c_int, set_zero_vector, sub_vectors, \
	angle_deg, norm
import numpy as np
from tqdm import tqdm

ctypedef struct possible_features:
	int chosen_angle
	int seedpoint
	int len


cdef void tracking(double[:,:,:,:] paths, double[:] seed,
                   int seed_shape, Interpolation interpolate,
              Integration integrate, Trafo trafo, Validator validator, int max_track_length,
	                   int samples, double[:,:,:,:] features, possible_features features_save) nogil except *:
	"""
        Initializes the tracking for one seed in both directions.
	@param paths:
	@param old_dir:
	@param seed:
	@param seed_shape:
	@param interpolate:
	@param integrate:
	@param trafo:
	@param validator:
	@param max_track_length:
	@param samples:
	@param features:
	"""
	cdef int k=0, j
	for j in range(samples):
		k=0
		while True:
			k+=1
			# set zero inclusion check
			set_zero_vector(validator.ROIIn.inclusion_check)
			if seed_shape == 3:
				interpolate.main_dir(paths[j, 0, 0])
				integrate.old_dir = interpolate.next_dir
			else:
				integrate.old_dir = seed[3:]
			status1 = forward_tracking(paths[j,:,0, :], interpolate, integrate, trafo, validator, max_track_length,
			                 features[j,:,0, :], features_save)
			if seed_shape == 3:
				interpolate.main_dir(paths[j, 0, 1])
				mult_with_scalar(integrate.old_dir, -1.0 ,interpolate.next_dir)
			else:
				mult_with_scalar(integrate.old_dir, -1.0 ,seed[3:])
			status2 = forward_tracking(paths[j,:,1,:], interpolate, integrate, trafo, validator, max_track_length,
			                 features[j,:,1, :], features_save)
			# if not found bot regions of interest delete path.
			if validator.ROIIn.included_checker() or not status1 or not status2:
				validator.set_path_zero(paths[j,:,1,:], features[j,:,1, :])
				validator.set_path_zero(paths[j, :, 0, :], features[j, :, 0, :])
				trafo.wtoi(seed[:3])
				paths[j, 0, 0] = trafo.point_wtoi
				paths[j, 0, 1] = trafo.point_wtoi
				if features_save.seedpoint:
					features[j, 0, 0, features_save.seedpoint] = 1
					features[j, 0, 1, features_save.seedpoint] = 1
			else:
				break
			if k==5:
				break

cdef bint forward_tracking(double[:,:] paths,  Interpolation interpolate,
                       Integration integrate, Trafo trafo, Validator validator, int max_track_length, double[:,:] features, possible_features feature_save) nogil except *:
	"""
        This function do the tracking into one direction.
	@param paths: empty path array to save the streamline points.
    @param interpolate: Interpolation object
    @param integrate: Integration object
    @param trafo: Trafo object
    @param validator: Validator object
	@param max_track_length:
    @param features: empty feature array. To save informations to the streamline
	"""
	# check wm volume
	cdef int k
	# thousand is max length for pathway
	interpolate.prob.old_fa = 1
	for k in range(max_track_length - 1):
		# validate indey and wm density.
		if validator.index_checker(paths[k]):
			set_zero_vector(paths[k])
			break
		if validator.wm_checker(paths[k]):
			trafo.itow(paths[k])
			paths[k] = trafo.point_itow
			break
		# find matching directions
		if sum_c(integrate.old_dir) == 0:
			trafo.itow(paths[k])
			paths[k] = trafo.point_itow
			break
		interpolate.interpolate(paths[k], integrate.old_dir, k)

		# Check next step is valid. If it is: Integrate. else break
		if validator.next_point_checker(interpolate.next_dir):
			set_zero_vector(paths[k])
			break
		integrate.integrate(interpolate.next_dir, paths[k])
		# update old dir
		paths[k+1] = integrate.next_point
		# check if next dir is near region of interest:

		trafo.itow(paths[k])
		paths[k] = trafo.point_itow
		features[k, 2] = validator.ROIIn.included(paths[k])
		if validator.ROIEx.excluded(paths[k]):
			return False
		if feature_save.chosen_angle:
			features[k,feature_save.chosen_angle] = interpolate.prob.chosen_angle
		# Check curvature between current point and point 30mm ago
		if validator.Curve.curvature_checker(paths[:k + 1], features[k:k+1,1]):
			return False
		integrate.old_dir = interpolate.next_dir
	else:
		trafo.itow(paths[k+1])
		paths[k+1] = trafo.point_itow
	if k == 0:
		trafo.itow(paths[k])
		paths[k] = trafo.point_itow
	return True
#double[:,:,:,:,:] vector_field, meta, double[:,:,:] wm_mask, double[:,:] seeds, integration,
#                   interpolation, prob, stepsize, double variance, int samples, int max_track_length, double wmmin,
#                   double expectation, verbose, logging, inclusion, exclusion, double max_angle, double[:,:] trafo_fsl,
#                   file
cpdef tracking_all(vector_field, wm_mask, seeds, tracking_parameters, postprocessing, ukf_parameters, logging, saving):
	"""
	@param vector_field: Array (4,3,x,y,z)
		Where the first dimension contains the length and direction, the second
		contains the directions.
	@param meta: Dictionary. Keys: space directions and space origin
		meta data of multivectorfield
	@param wm_mask: Array (x,y,z)
		WM Mask to check wm density
	@param seeds: Array (3,r) or (6,r)
		seedpoint array. If first first axis has length 3 only seed point is given.
		If 6 also the initial direcition is given.
	@param integration: String
		Only Euler integration is available.
	@param interpolation: String
		FACT und Trilinear are available
	@param prob: String
		Gaussian, Laplacian, ScalarOld and ScalarNew are implemented
	@param stepsize: double
		Stepsize for Euler integration.
	@param variance: double
		Variance of Probabilistic Selection Method.
	@param samples: int
		Count of samples per seed.
	@param max_track_length: int
		Maximal length of each track.
	@param wmmin: double
		Minimal White Matter density
	@param expectation: double
		Expectation of Probabilistic selection method.
	@return:
	"""
	cdef Interpolation interpolate
	cdef Integration integrate
	cdef Trafo trafo
	cdef Probabilities directionGetter
	cdef Validator validator
	#select appropriate model
	if tracking_parameters['prob'] == "Gaussian":
		directionGetter = Gaussian(0, tracking_parameters['variance'])
	elif tracking_parameters['prob'] == "Laplacian":
		directionGetter = Laplacian(0, tracking_parameters['variance'])
	elif tracking_parameters['prob'] == "ScalarOld":
		directionGetter = ScalarOld(tracking_parameters['expectation'], tracking_parameters['variance'])
	elif tracking_parameters['prob'] == "ScalarNew":
		directionGetter = ScalarNew(tracking_parameters['expectation'], tracking_parameters['variance'])
	elif tracking_parameters['prob'] == "Deterministic":
		directionGetter = Deterministic(tracking_parameters['expectation'], tracking_parameters['variance'])
	elif tracking_parameters['prob'] == "Deterministic2":
		directionGetter = Deterministic2(tracking_parameters['expectation'], tracking_parameters['variance'])
	else:
		logging.error('Gaussian or Laplacian or Scalar are available so far. ')
		return 0

	trafo = <Trafo> Trafo(np.float64(tracking_parameters['space directions']), np.float64(tracking_parameters['space origin']))
	trafo_matrix = np.zeros((4,4))
	trafo_matrix[:3,:3] = tracking_parameters['space directions']
	trafo_matrix[:3,3] = tracking_parameters['space origin']
	trafo_matrix[3,3] = 1
	validator = Validator(wm_mask,np.array(wm_mask.shape, dtype=np.intc), tracking_parameters['wmmin'], postprocessing['inclusion'], postprocessing['exlusion'], tracking_parameters['max_angle'], trafo,
	                      tracking_parameters['trafo_fsl'])
	#cdef Integration integrate
	if tracking_parameters['integration'] == "Euler":
		integrate = Euler(tracking_parameters['space directions'], tracking_parameters['space origin'], trafo, float(tracking_parameters['stepsize']))
	else:
		logging.error('Only Euler is available so far. Hence set Euler as argument.')
		return 0

	cdef int[:] dim = np.array(vector_field.shape, dtype=np.int32)
	if tracking_parameters['interpolation'] == "FACT":
		interpolate = FACT(vector_field, dim[2:5], directionGetter)
	elif tracking_parameters['interpolation'] == "Trilinear":
		interpolate = Trilinear(vector_field, dim[2:5], directionGetter)
	elif tracking_parameters['interpolation'] == "UKF MultiTensor":
		interpolate = UKFMultiTensor(vector_field, dim[2:5], directionGetter, ukf_parameters)
	elif tracking_parameters['interpolation'] == "UKF low rank":
		interpolate = UKFFodf(vector_field, dim[2:5], directionGetter, ukf_parameters)
	else:
		logging.error('FACT, Triliniear or UKF for MultiTensor and low rank approximation are available so far.')
		return 0
	cdef int i, j, k, m = tracking_parameters['seeds'].shape[0]
	# Array to save Polygons
	cdef double[:,:,:,:,:] paths = np.zeros((1 if saving['file'] else m, tracking_parameters['samples'], tracking_parameters['max_track_length'], 2, 3),dtype=np.float64)
	# Array to save features belonging to polygons
	cdef double[:,:,:,:,:] features = np.zeros((1 if saving['file'] else m, tracking_parameters['samples'], tracking_parameters['max_track_length'], 2, saving['features']['len']),dtype=np.float64)
	# loop through all seeds.
	tracks = []
	tracks_len = []
	k = 0
	for i in tqdm(range(m), disable=not tracking_parameters['verbose']):
		#k = 0 if saving['file'] else k+=1
		if saving['file']:
			k = 0
		else: k = k +1
		#Convert seedpoint
		trafo.wtoi(seeds[i][:3])
		for j in range(tracking_parameters['samples']):
			validator.set_path_zero(paths[k,j, :, 1, :], features[k,j, :, 1, :])
			validator.set_path_zero(paths[k,j, :, 0, :], features[k,j, :, 0, :])
			paths[k,j, 0, 0] = trafo.point_wtoi
			paths[k,j, 0, 1] = trafo.point_wtoi
			if "Deterministic" in tracking_parameters['prob']:
				for k in range(3):
					paths[k,j, 0, 0,k] +=  np.random.normal(0,1,1)
					paths[k,j, 0, 1,k] = paths[k,j, 0, 0,k]

		if saving['features']['seedpoint']:
			features[k,:, 0, 0, saving['features']['seedpoint']] = 1
			features[k,:, 0, 1, saving['features']['seedpoint']] = 1
		#Do the tracking for this seed with the direction
		tracking(paths[k], seeds[i], seeds[i].shape[0], interpolate, integrate, trafo, validator, tracking_parameters['max_track_length'], tracking_parameters['samples'], features[k], saving['features'])
		# delete all zero arrays.
		for j in range(tracking_parameters['samples']):
			path = np.concatenate((np.asarray(paths[k,j]),np.asarray(features[k,j])), axis=-1)
			path = np.concatenate((path[:,0][::-1], path[:,1]))
			to_exclude = np.all(path[:,:4] == 0, axis=1)
			path = path[~to_exclude]
			if sum_c(path[:,3]) == 2:
				path = np.delete(path, np.argwhere(path[:,3]==1)[0], axis=0)
			if path.shape[0]>5:
				# Work on disk or ram. Ram might be faster but for large files disk is preferable.
				if saving['file']:
					with open(saving['file'] + 'len', 'a') as f:
						f.write(str(path.shape[0]) +'\n')
					with open(saving['file'], 'a') as f:
						for i in range(path.shape[0]):
							f.write(' '.join(map(str, path[i])) + "\n")
				else:
					tracks_len.append(path.shape[0])
					tracks += [tuple(x) for x in path]

	return tracks, tracks_len


