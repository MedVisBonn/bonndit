#%%cython --annotate
#cython: language_level=3, boundscheck=False,


import numpy as np
import cython
from cython.parallel cimport prange, threadid
import psutil
from bonndit.utilc.cython_helpers cimport angle_deg, add_vectors, mult_with_scalar, set_zero_vector, norm, sum_c
from libc.math cimport fabs

c = [(x,y,z) for x in (1,2,3) for y in (4,5,0) for z in (7,0)]
d =[(x,y,z) for x in c for y in c for z in c if sum(set(x + y+ z)) == 22 and 22 == sum(x) + sum(y) + sum(z)]
cdef int[:,:,:] all_opt = np.array([d[x] for x in range(len(d)) if not [y for y in d[:x] if (d[x][0] in y) and (d[x][1] in y) and (d[x][2] in y)]], dtype=np.int32)

cdef void mean_calc(double[:,:] output, double[:,:,:] vectors, double[:] prob) except *:
	"""
	Given the three deconv with 1,2 and three fibers, this script groups them into three groups where each group
	contains at most one fiber out of a model. The groups are build by minimizing the summed average. To generate
	this all combinations (18) are tested and the minimum is chosen. Then the weighted average vector out of each
	group is calculated and returned.

	Parameters
	----------
	prob    probability to build the weighted average.
	output  Set of three vectors which is generated.
	vectors 6 input vectors which are matched to three groups

	Returns
	-------


	"""
	cdef int i,j,k, l, best_ind = 0
	cdef double min_var = 0, angle_sum = 0, to_norm=0
	cdef double[:,:] vec = np.zeros((3,3), dtype=np.float64)
	cdef double[:] avg_vec = np.zeros((3,), dtype=np.float64)
	# Loop through all possibilities
	for i in range(18):
		angle_sum = 0
		# calculate angle sum for each possibility
		for j in range(3):
			for k in range(3):
				set_zero_vector(vec[k])
				# Select correct vectors
				if all_opt[i, j, k] != 0:
					mult_with_scalar(vec[k], 1, vectors[(all_opt[i, j, k] -1)// 3, 1:, (all_opt[i,j,k] - 1) % 3])
					if sum_c(vec[k]) == 0:
						print('0 again', vectors[0])
			set_zero_vector(avg_vec)
			# add vectors groupwise
			for k in range(3):
				if all_opt[i, j, k] != 0:
					if sum_c(avg_vec) != 0:
						test_angle = angle_deg(avg_vec, vec[k])
						if test_angle > 90:
							mult_with_scalar(vec[k], -1, vec[k])
					add_vectors(avg_vec, avg_vec, vec[k])
			# calc angle between avg_vec and all group members.
			for k in range(3):
				if all_opt[i, j, k] != 0:
					angle_sum += fabs(angle_deg(avg_vec, vec[k]))
		# save index with lowest angle sum.

		if angle_sum < min_var or i == 0:
			min_var = angle_sum
			best_ind = i
	# Take the best index and return it.
	for i in range(3):
		l = 0
		# renom the probabilities to avoid, that the second and third vector are short.
		to_norm=0
		#for j in range(3):
		#	if all_opt[best_ind, i, j] == 0:
		#		l += 1
		#		continue
		#	to_norm += prob[(all_opt[best_ind, i, j] - 1) // 3]

		for j in range(3):
			if all_opt[best_ind, i, j] == 0:
				l += 1
				continue
			mult_with_scalar(vec[j], prob[(all_opt[best_ind, i, j] - 1) // 3],vectors[(all_opt[best_ind, i,
			                                                                                   j] - 1) // 3, 1:, (all_opt[best_ind, i, j] - 1) % 3])

		#	mult_with_scalar(vec[j], prob[(all_opt[best_ind, i, j] - 1) // 3]
		#	                 * abs(vectors[(all_opt[best_ind, i,j] -1)// 3, 0, (all_opt[best_ind,i,j] - 1) % 3]),
			#	                 vectors[(all_opt[best_ind, i,j] -1)// 3, 1:,(all_opt[best_ind,i,j] - 1) % 3])
			if sum_c(output[1:, i]) != 0:
				if fabs(angle_deg(output[1:, i], vec[j])) > 90:
					mult_with_scalar(vec[j], -1, vec[j])
			add_vectors(output[1:, i], output[1:, i], vec[j])
			to_norm += prob[(all_opt[best_ind, i, j] - 1) // 3]*fabs(vectors[(all_opt[best_ind, i,j] -1)// 3, 0,
			                                                             (all_opt[best_ind,i,j] - 1) % 3])


		output[0, i] = to_norm
		if norm(output[1:,i])  != 0:
			mult_with_scalar(output[1:, i], 1/norm(output[1:,i]) , output[1:, i])







