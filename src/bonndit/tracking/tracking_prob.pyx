#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True
import sys
sys.path.append('.')
from .alignedDirection cimport  Gaussian, Laplacian, ScalarOld, ScalarNew, Probabilities
from .ItoW cimport Trafo
from .integration cimport  Euler, Integration
from .interpolation cimport  FACT, Trilinear, Interpolation
from helper_functions.cython_helpers cimport mult_with_scalar, sum_c, set_zero_matrix, set_zero_vector, sub_vectors, \
	angle_deg, norm
import psutil
import numpy as np
from cython.parallel cimport prange, threadid

cdef void tracking(double[:,:,:,:] paths, double[:,:,:] wm_mask, double[:] old_dir, double[:] seed,
                   int seed_shape, Interpolation interpolate,
              Integration integrate, Trafo trafo, int samples, int max_track_length, double wmmin, double[:,:,:,:] features) nogil except *:
	cdef int j
	#500 random paths
	for j in range(samples):
		#print(j)
		if seed_shape == 3:
			interpolate.main_dir(paths[0, 0, 1])
			old_dir = interpolate.next_dir
		else:
			old_dir = seed[3:]
		#print(j)
		#with gil:
	#		print(*interpolate.next_dir)
		forward_tracking(paths[j,:,0, :], wm_mask, old_dir, interpolate, integrate, trafo, max_track_length, wmmin, features[j,:,0, :])
		if seed_shape == 3:
			interpolate.main_dir(paths[0, 0, 1])
			mult_with_scalar(old_dir, -1.0 ,interpolate.next_dir)
		else:
			mult_with_scalar(old_dir, -1.0 ,seed[3:])
		#with gil:
		#	print(*interpolate.next_dir)
		#print(*paths[j,0,1])
		forward_tracking(paths[j,:,1,:], wm_mask, old_dir, interpolate, integrate, trafo, max_track_length, wmmin,
		                 features[j,:,1, :])


cdef void forward_tracking(double[:,:] paths, double[:,:,:] wm_mask, double[:] old_dir,  Interpolation interpolate,
                       Integration integrate, Trafo trafo, int max_track_length, double wmmin, double[:,:] features) \
		nogil except *:
	# check wm volume
	cdef int k, l
	# thousand is max length for pathway
	interpolate.prob.old_fa = 1
#	cdef double sum_angle = 0
	for k in range(max_track_length - 1):
		#break if white matter is to low.
		if wm_mask[int(paths[k,0]), int(paths[k,1]), int(paths[k,2])] < wmmin:
			if k<10:
				set_zero_matrix(paths)
				set_zero_matrix(features)
			break
		# find matching directions
		if sum_c(old_dir) == 0:
			break
		interpolate.interpolate(paths[k], old_dir)

		# Check next step is valid. If it is: Integrate. else break
		if sum_c(interpolate.next_dir) == 0 or sum_c(interpolate.next_dir) != sum_c(interpolate.next_dir):
			if k<10:
				set_zero_matrix(paths)
				set_zero_matrix(features)
			break
		else:
			integrate.integrate(interpolate.next_dir, paths[k])

		# update old dir
		paths[k+1] = integrate.next_point
		# Check curvature between current point and point 30mm ago
		for l in range(min(30,k)): #max(1, min(30,k)) -
			sub_vectors(paths[k + 2], paths[k - l], paths[k - l + 1])
			sub_vectors(paths[k + 3], paths[k], paths[k + 1])
			if sum_c(paths[k + 2]) != 0 and sum_c(paths[k + 3])!=0:
				features[k,1] = angle_deg(paths[k + 2], paths[k + 3])

			if features[k,1] > 120:
				#pass
				set_zero_matrix(paths)
				set_zero_matrix(features)
				break
			else:
				set_zero_matrix(paths[k + 2: k + 4])

		#	sub_vectors(paths[k + 2], paths[k - 29], paths[k+1])
		#	features[k, 3] = norm(paths[k + 2])
			#if features[k, 3] < 20:
			#	set_zero_matrix(paths)
			#	set_zero_matrix(features)
			#	break
		#	set_zero_vector(paths[k + 2])

	#	features[k, 1] = interpolate.prob.chosen_angle
	#	features[k,4] = interpolate.prob.chosen_prob
		old_dir = interpolate.next_dir
#	with gil:
#		print(sum_angle)




cpdef tracking_all(double[:,:,:,:,:] vector_field, meta, double[:,:,:] wm_mask, double[:,:] seeds, integration,
                   interpolation, prob, stepsize, double variance, int samples, int max_track_length, double wmmin,
                   double expectation):
	"""

	Parameters
	----------
	vector_field
	meta
	wm_mask
	seeds
	integration
	interpolation
	prob
	stepsize
	variance
	samples
	max_track_length
	wmmin
	expectation

	Returns
	-------

	"""
	#print(1)
	cdef Interpolation interpolate
	#cdef (Interpolationcd) interpolate
	cdef Integration integrate
	cdef Trafo trafo
	cdef Probabilities directionGetter

	#select appropriate model
	#cdef Probabilities directionGetter
	if prob == "Gaussian":
		directionGetter = Gaussian(0, variance)
	elif prob == "Laplacian":
		directionGetter = Laplacian(0, variance)
	elif prob == "ScalarOld":
		directionGetter = ScalarOld(expectation, variance)
	elif prob == "ScalarNew":
		directionGetter = ScalarNew(expectation, variance)
	else:
		print('Gaussian or Laplacian or Scalar are available so far. ')
		return 0


	trafo = <Trafo> Trafo(np.float64(meta['space directions']), np.float64(meta['space origin']))
	#cdef Integration integrate
	if integration == "Euler":
		integrate = Euler(meta['space directions'], meta['space origin'], trafo, float(stepsize))
#	elif integration == 'FACT':
#		integrate = integration.FACT(meta['space directions'][2:], meta['space origin'], trafo, float(args.stepsize))
#	elif integration == 'rungekutta':
#		integrate = integration.RungeKutta(meta['space directions'][2:], meta['space origin'], trafo,
#		                                   float(stepsize), interpolate)
	else:
		print('Only Euler is available so far. Hence set Euler as argument.')
		return 0

	#cdef Interpolation interpolate
	# Trilinear interpolation possible more options
	cdef int[:] dim = np.array(vector_field.shape, dtype=np.int32)
	#print(np.asarray(dim))
	if interpolation == "FACT":
		interpolate = FACT(vector_field, dim[2:5], trafo, directionGetter)
	elif interpolation == "Trilinear":
		interpolate = Trilinear(vector_field, dim[2:5], trafo, directionGetter)
	# Integration options euler and FACT
	else:
		print('FACT or Triliniear are available so far.')
		return 0
	cdef int i, j, k, m = seeds.shape[0]
	cdef double[:] old_dir = np.ndarray((3,))
	# Array to save Polygons
	cdef double[:,:,:,:,:] paths = np.zeros((m, samples, max_track_length, 2, 3),dtype=np.float64)
	# Array to save features belonging to polygons
	cdef double[:,:,:,:,:] features = np.zeros((m, samples, max_track_length, 2, 2),dtype=np.float64)
#	print(paths.shape)
	# loop through all seeds.
	tracks = []
	tracks_len = []

	for i in range(m):

		#Convert seedpoint
		trafo.wtoi(seeds[i][:3])
		for j in range(samples):
			paths[i, j, 0, 0] = trafo.point_wtoi
			paths[i, j, 0, 1] = trafo.point_wtoi
		features[i, :, 0, 0, 0] = 1
		features[i, :, 0, 1, 0] = 1
		#with gil:
		#Select main direction from nearest neighbor

		#print(i)
		#Do the tracking for this seed with the direction
		tracking(paths[i], wm_mask, old_dir, seeds[i], seeds[i].shape[0], interpolate, integrate, trafo,
		         samples, max_track_length, wmmin, features[i])
		#trafo.wtoi(seeds[i][:3])


	for i in range(m):
		for j in range(samples):
			for k in range(max_track_length):
				if sum_c(paths[i,j,k,0]) != 0:
					trafo.itow(paths[i,j,k,0])
					paths[i,j,k,0] = trafo.point_itow
				if sum_c(paths[i,j,k,1]) != 0:
					trafo.itow(paths[i, j, k, 1])
					paths[i, j, k, 1] = trafo.point_itow
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

