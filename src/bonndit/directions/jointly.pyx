import numpy as np
from bonndit.utilc.hota import hota_4o3d_sym_s_form, hota_4o3d_sym_two_vec, hota_4o3d_sym_v_form
from bonndit.utilc.cython_helpers import scalar
from scipy.optimize import NonlinearConstraint, Bounds, minimize


### TODO: doublecheck $\| v \| = 1$.
#	TODO: update this afterwards.
class Optimize:
	def __init__(self, x, tensor, mean_l, mean_v, neigh, rank=3, nu=1):
		self.neigh = neigh  # i
		self.rank = rank  # j
		self.old_loss = 0
		self.nu = nu
		self.loss = 0
		self.tensor = tensor
		self.mean_l = mean_l
		self.mean_v = mean_v
		self.x = np.array(x)
		self.grad = np.zeros(x.shape)

	def norm(self, x):
		constrains = []
		for i in range(self.neigh):
			skip = i * self.rank * 4
			for j in range(self.rank):
				constrains.append(NonlinearConstraint(np.linalg.norm(x[4 * j + 1 + skip: 4 * j + 4 + skip]) ** 2),1,1)


	def calc_sum(self, current_neigh):
		ret = 0
		for i in range(self.rank):
			for j in range(i + 1, self.rank):
				ret += self.x[4 * i + self.rank * current_neigh * 4] * self.x[
					4 * j + self.rank * current_neigh * 4] * hota_4o3d_sym_two_vec(
					self.x[4 * j + self.rank * current_neigh * 4 + 1: 4 * j + self.rank * current_neigh * 4 + 4],
					self.x[4 * i + self.rank * current_neigh * 4 + 1: 4 * i + self.rank * current_neigh * 4 + 4])
		return ret

	def calc_sum_2(self, current_neigh, skip):
		ret = [0]*3
		for i in range(self.rank):
			if i == skip:
				continue
			ret += self.x[4 * i + self.rank * current_neigh * 4] * hota_4o3d_sym_two_vec(
				self.x[4 * skip + self.rank * current_neigh * 4 + 1: 4 * skip + self.rank * current_neigh * 4 + 4],
				self.x[4 * i + self.rank * current_neigh * 4 + 1: 4 * i + self.rank * current_neigh * 4 + 4])
		for i in range(self.rank):
			ret[i] *= self.x[4 * skip + self.rank * current_neigh * 4]
		return ret

	#def loss(self):
	#	# save old loss to check progress
	#	self.old_loss = self.loss
	#	self.loss = 0
#
	#	for i in range(self.neigh):
	#		skip = self.rank * i * 4
	#		for j in range(self.rank):
	#			first = 2 * self.x[4 * j + skip] * hota_4o3d_sym_s_form(self.tensor,
	#																	self.x[4 * j + skip + 1: 4 * j + skip + 4])
	#			second = 2 * self.calc_sum(i)
	#			third = 2 * self.nu * self.mean_l[i * self.rank + j] * self.x[4 * j + skip] * scalar(
	#				self.mean_v[i * self.rank * 3 + 3 * j, i * self.rank * 3 + 3 * (j + 1)],
	#				self.x[4 * j + skip + 1: 4 * j + skip + 4])
	#			fourth = (1 + self.nu) * self.x[4 * j + skip] ** 2
	#			fifth = self.x[4 * j + skip + 1: 4 * j + skip + 4](
	#				scalar(self.x[4 * j + skip + 1: 4 * j + skip + 4], self.x[4 * j + skip + 1: 4 * j + skip + 4]) - 1)
	#			self.loss -= first + second - third + fourth + fifth
	#	return self.loss

	def loss(self):

		for i in range(self.neigh):
			skip = self.rank * i * 4
			for j in range(self.rank):
				self.grad[skip * j * 4] = (2 * (hota_4o3d_sym_s_form(self.tensor, self.x[4 * j + skip + 1: 4 * j + skip + 4])+ self.calc_sum_2(i,j) + self.nu *self.mean_l[i * self.rank + j] *scalar(self.mean_v[i * self.rank * 3 + 3 * j, i * self.rank * 3 + 3 * (j + 1)], self.x[4 * j + skip + 1: 4 * j + skip + 4]) + (1 + self.nu) * self.x[4 * j + skip]))**2
				hota_4o3d_sym_v_form(first, self.tensor, self.x[4 * j + skip + 1: 4 * j + skip + 4])
				first *= 8
				second = 8 * self.calc_sum_2(i, j)
				third = 2 * self.nu * self.mean_l[i * self.rank + j] * self.x[4 * j + skip: 4 * j + skip + self.rank * 4: self.rank] * self.mean_v[i * self.rank * 3 + 3 * j, i * self.rank * 3 + 3 * (j + 1)]
				fourth = 2 * self.x[4 * j + skip + 1: 4 * j + skip + 4] * self.x[4 * j + skip + 4]
				self.grad[skip * j * 4 + 1: skip * j * 4 + 4] = (first + second + third + fourth)**2
				self.grad[skip * j * 4 + 4] = (scalar(self.x[4 * j + skip + 1: 4 * j + skip + 4], self.x[4 * j + skip + 1: 4 * j + skip + 4]) - 1)**2



	def optimization(self):
		minimize()





