#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import sys
sys.path.append('.')
from .alignedDirection cimport  Gaussian, Laplacian, ScalarOld, ScalarNew, Probabilities
from .ItoW cimport Trafo
from .stopping cimport Validator
from .integration cimport  Euler, Integration
from .interpolation cimport  FACT, Trilinear, Interpolation
from bonndit.utilc.cython_helpers cimport mult_with_scalar, sum_c, set_zero_matrix, set_zero_vector, sub_vectors, \
	angle_deg, norm
import numpy as np
from tqdm import tqdm


cdef void tracking(double[:,:,:,:] paths, double[:] seed,
                   int seed_shape, Interpolation interpolate,
              Integration integrate, Trafo trafo, Validator validator, int max_track_length,
	                   int samples, double[:,:,:,:] features) nogil except *:
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
	cdef int j
	for j in range(samples):
		# set zero inclusion check
		set_zero_vector(validator.ROI.inclusion_check)
		if seed_shape == 3:
			interpolate.main_dir(paths[j, 0, 0])
			integrate.old_dir = interpolate.next_dir
		else:
			integrate.old_dir = seed[3:]
		forward_tracking(paths[j,:,0, :], interpolate, integrate, trafo, validator, max_track_length,
		                 features[j,:,0, :])
		if seed_shape == 3:
			interpolate.main_dir(paths[j, 0, 1])
			mult_with_scalar(integrate.old_dir, -1.0 ,interpolate.next_dir)
		else:
			mult_with_scalar(integrate.old_dir, -1.0 ,seed[3:])
		forward_tracking(paths[j,:,1,:], interpolate, integrate, trafo, validator, max_track_length,
		                 features[j,:,1, :])
		# if not found bot regions of interest delete path.
		if validator.ROI.included_checker():
			validator.set_path_zero(paths[j,:,1,:], features[j,:,1, :])
			validator.set_path_zero(paths[j, :, 0, :], features[j, :, 0, :])

cdef void forward_tracking(double[:,:] paths,  Interpolation interpolate,
                       Integration integrate, Trafo trafo, Validator validator, int max_track_length, double[:,:] features) nogil except *:
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
	cdef int k, l
	# thousand is max length for pathway
	interpolate.prob.old_fa = 1
	for k in range(max_track_length - 1):
		# validate indey and wm density.
		if validator.index_checker(paths[k]):
			break
		if validator.wm_checker(paths[k]):
			break
		# find matching directions
		if sum_c(integrate.old_dir) == 0:
			break
		interpolate.interpolate(paths[k], integrate.old_dir)

		# Check next step is valid. If it is: Integrate. else break
		if validator.next_point_checker(interpolate.next_dir):
			break

		integrate.integrate(interpolate.next_dir, paths[k])
		# update old dir
		paths[k+1] = integrate.next_point
		# check if next dir is near region of interest:

		trafo.itow(paths[k])
		paths[k] = trafo.point_itow
		validator.ROI.included(paths[k])
		# Check curvature between current point and point 30mm ago
		if validator.Curve.curvature_checker(paths[:k + 1], features[k:k+1,1]):
			validator.set_path_zero(paths, features)
			return
		integrate.old_dir = interpolate.next_dir
	if k == 0:
		trafo.itow(paths[k])
		paths[k] = trafo.point_itow
	if k == max_track_length - 2:
		if sum_c(paths[k+1]) == 0:
			with gil:
				print('1 Fehler')
		trafo.itow(paths[k+1])
		paths[k+1] = trafo.point_itow
	else:
		if sum_c(paths[k]) == 0:
			with gil:
				print('Ne oder')
		trafo.itow(paths[k])
		paths[k] = trafo.point_itow






cpdef tracking_all(double[:,:,:,:,:] vector_field, meta, double[:,:,:] wm_mask, double[:,:] seeds, integration,
                   interpolation, prob, stepsize, double variance, int samples, int max_track_length, double wmmin,
                   double expectation, verbose, logging, inclusion, double max_angle, double[:,:] trafo_fsl):
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
	if prob == "Gaussian":
		directionGetter = Gaussian(0, variance)
	elif prob == "Laplacian":
		directionGetter = Laplacian(0, variance)
	elif prob == "ScalarOld":
		directionGetter = ScalarOld(expectation, variance)
	elif prob == "ScalarNew":
		directionGetter = ScalarNew(expectation, variance)
	else:
		logging.error('Gaussian or Laplacian or Scalar are available so far. ')
		return 0

	trafo = <Trafo> Trafo(np.float64(meta['space directions'][2:]), np.float64(meta['space origin']))
	trafo_matrix = np.zeros((4,4))
	trafo_matrix[:3,:3] = meta['space directions'][2:]
	trafo_matrix[:3,3] = meta['space origin']
	trafo_matrix[3,3] = 1
	validator = Validator(wm_mask,np.array(wm_mask.shape, dtype=np.intc), wmmin, inclusion, max_angle, trafo, trafo_matrix, trafo_fsl)
	#cdef Integration integrate
	if integration == "Euler":
		integrate = Euler(meta['space directions'][2:], meta['space origin'], trafo, float(stepsize))
	else:
		logging.error('Only Euler is available so far. Hence set Euler as argument.')
		return 0

	cdef int[:] dim = np.array(vector_field.shape, dtype=np.int32)
	if interpolation == "FACT":
		interpolate = FACT(vector_field, dim[2:5], trafo, directionGetter)
	elif interpolation == "Trilinear":
		interpolate = Trilinear(vector_field, dim[2:5], trafo, directionGetter)
	# Integration options euler and FACT
	else:
		logging.error('FACT or Triliniear are available so far.')
		return 0
	cdef int i, j, k, m = seeds.shape[0]
	# Array to save Polygons
	cdef double[:,:,:,:,:] paths = np.zeros((m, samples, max_track_length, 2, 3),dtype=np.float64)
	# Array to save features belonging to polygons
	cdef double[:,:,:,:,:] features = np.zeros((m, samples, max_track_length, 2, 2),dtype=np.float64)
	# loop through all seeds.
	tracks = []
	tracks_len = []

	for i in tqdm(range(m), disable=not verbose):
		#Convert seedpoint
		trafo.wtoi(seeds[i][:3])
		for j in range(samples):
			paths[i, j, 0, 0] = trafo.point_wtoi
			paths[i, j, 0, 1] = trafo.point_wtoi
		features[i, :, 0, 0, 0] = 1
		features[i, :, 0, 1, 0] = 1
		#Do the tracking for this seed with the direction
		tracking(paths[i], seeds[i], seeds[i].shape[0], interpolate, integrate, trafo, validator,
		         max_track_length, samples, features[i])
		# delete all zero arrays.
		for j in range(samples):
			path = np.concatenate((np.asarray(paths[i,j]),np.asarray(features[i,j])), axis=-1)
			path = np.concatenate((path[:,0][::-1], path[:,1]))
			to_exclude = np.all(path[:,:4] == 0, axis=1)
			path = path[~to_exclude]
			if sum_c(path[:,3]) == 2:
				path = np.delete(path, np.argwhere(path[:,3]==1)[0], axis=0)
			if path.shape[0]>5:
				tracks_len.append(path.shape[0])
				tracks += [tuple(x) for x in path]

	return tracks, tracks_len


