#%%cython --annotate
#cython: language_level=3, boundscheck=True, wraparound=True


import numpy as np
import cython
from bonndit.utilc.cython_helpers cimport mult_with_scalar, argmax, sub_vectors, norm, sum_c
from bonndit.utilc.hota cimport  hota_4o3d_sym_eval
from .means cimport mean_calc
from libc.math cimport log, exp, pow
from cython.parallel cimport prange, threadid
import psutil
from tqdm import tqdm


# Multiplier of the tensor entries.
cdef int[:] MULTIPLIER = np.array([1.0, 1.0, 1.0, 1.0,1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 6.0, 3.0, 1.0,
                                3.0, 3.0, 1.0, 1.0, 4.0, 4.0, 6.0, 12.0, 6.0, 4.0, 12.0, 12.0, 4.0, 1.0, 4.0, 6.0,
                             4.0, 1.0], dtype=np.int32)

# Length mapping to index
cdef int[:] LENGTH =  np.array([1,3,6,10,15], dtype=np.int32)


cpdef void model_avg(double[:,:,:,:,:] output, double[:,:,:,:,:,:] vectorfields, double[:,:,:,:] fodf,  str model,
                     double[:,:,:,:] prob, double x, double y, verbose)  except *:
	"""
	Given three multi vectorfields with one, two and three fibers per voxel. This script calculates
		the weighted average voxelwise. With weights build from the probabilities of the deconv.
	or
		the chooses the model with the max probability.

	Parameters
	----------
	y
	x
	prob
	model:           'selection' or 'average'
	output:          Output vectorfield
	vectorfields:    three multivectorfields.
	fodf:            fodfs which where used to generate the low rank approx.
	res:
	verbose

	Returns
	-------


	"""
	cdef int i, j, k, index, dim0 = vectorfields.shape[3], dim1 = vectorfields.shape[4], dim2 = vectorfields.shape[5]
	cdef int thread_num = psutil.cpu_count()
	cdef double[:,:] res = np.zeros((3,15))
	cdef double[:,:,:] mean_ary = np.zeros((thread_num, 4,3))
	# Loop through each voxel
	for i in tqdm(range(dim0), disable=not verbose):
		for j in range(dim1):
			for k in range(dim2):
				if fodf[0,i,j,k] < 0.5:
					continue
				# Calculate the probability of each model
				res = calc_res(fodf[1:,i,j,k], vectorfields[:,:,:,i,j,k])
				prob[:,i,j,k] = calc_prob(fodf[1:,i,j,k], res,x,y)

				if sum(prob[:, i, j, k]) != 0:
					mult_with_scalar(prob[:, i, j, k], 1 / (prob[0, i, j, k] + prob[1, i, j, k] + prob[2, i, j, k]),
					                 prob[:, i, j, k])

				if model == 'selection':
					# If selection, take the model with the highest prob. And write the model vectors to the output
					index = argmax(prob[:,i,j,k])
					for l in range(3):
						mult_with_scalar(output[:,l, i, j, k], 1, vectorfields[index, :, l, i, j, k])
				elif model == 'averaging':
					# Build three groups of vectors which are most aligned and multiply them with the weights to get
					# the ouput

					mean_calc(output[:,:,i,j,k], vectorfields[:,:,:,i, j,k], prob[:,i,j,k])
				else:
					raise ValueError("Argument model must be either 'selection' or 'averaging'")



cdef double tensor_norm(double[:] a)  except *:
	""" Calculates tensor norm of a given tensor up to rank 4. Select first the multiplier out of a given set of
	multipliers and multiply them with the tensor entry.

	Parameters
	----------
	a: Tensor

	Returns
	-------
	Tensor norm

	"""
	# Select range of matching multiplier.
	cdef int start_ind = 0, l = a.shape[0], i
	cdef double res = 0
	for i in range(5):
		if LENGTH[i] == l:
			break
		start_ind += LENGTH[i]

	for i in range(l):
		res += a[i]*a[i]*MULTIPLIER[start_ind + i]
	return pow(res, 1/2)


cdef double kumaraswamy_pdf(double x, double a, double b)  except *:
	return a * b * (pow(x, (a - 1))) * pow((1 - pow(x ,a)),(b - 1))

cdef double[:] expectation = np.array([1.0,1.0,1.0]) #[0.40,0.44,0.16])

cdef double[:] calc_prob(double[:] fodf, double[:,:] res, double x, double y) except *:
	""" Ref Gemmas' Masterthesis. Calculate the Probabilities of each model according to gemmas masterthesis.

	Parameters
	----------
	fodf : Rank 4 Tensor
	res : Rank 4 Error Tensor

	Returns
	-------
	Probability of each model.
	"""
	cdef double norm_fodf, res_norm, bic
	cdef double[:] prob = np.zeros((3,)), ratio = np.zeros((3,))
	norm_fodf =  tensor_norm(fodf)
	for i in range(3):
		res_norm = tensor_norm(res[i])
		ratio[i] = res_norm/norm_fodf
		bic = 3*(3 - i)*log(15) - 2 * log(kumaraswamy_pdf(ratio[i], x, y))
		prob[i] = exp(-bic/2)*expectation[i]
	#if sum_c(prob):
	#	mult_with_scalar(prob, 1/(prob[0] + prob[1] + prob[2]), prob)
	#	with open('test.csv', 'a') as f:
	#		f.write(','.join(map(str, list(prob))) + '\n')
	return prob




cdef double[:,:] calc_res(double[:] fodf, double[:,:,:] vectorfields) except *:
	"""
	Calculates the residual of all deconv.
	Parameters
	----------
	fodf
	vectorfields

	Returns
	-------

	"""
	cdef double[:,:] output = np.array([fodf, fodf, fodf])
	cdef double[:] res = np.zeros((15,))
	for i in range(3):
		for j in range(3-i):
			hota_4o3d_sym_eval(res, vectorfields[i, 0, j], vectorfields[i, 1:, j])
			sub_vectors(output[i], output[i], res)
	return output
