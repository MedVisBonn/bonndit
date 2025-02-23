#%%cython --annotate
#cython: language_level=3, boundscheck=True, wraparound=False, warn.unused=True, warn.unused_args=True, profile=False,
# warn.unused_results=True, cython: profile=True
import Cython
import cython
import torch
from bonndit.directions.fodfapprox cimport approx_all_spherical
from tqdm import tqdm
from scipy.optimize import nnls
from bonndit.utilc.cython_helpers cimport add_pointwise, floor_pointwise_matrix, norm, mult_with_scalar,\
    add_vectors, sub_vectors, scalar, clip, set_zero_vector, set_zero_matrix, sum_c, sum_c_int, set_zero_vector_int, \
    angle_deg, set_zero_matrix_int, point_validator, cart2sphere, cross
import numpy as np
import time
from bonndit.utilc.cython_helpers cimport dm2toc
from bonndit.utilc.hota cimport hota_6o3d_sym_norm
from bonndit.utils.esh import esh_to_sym_matrix, sym_to_esh_matrix
from bonndit.utilc.hota cimport hota_4o3d_sym_norm, hota_4o3d_sym_eval, hota_8o3d_sym_eval, hota_6o3d_sym_eval,hota_6o3d_sym_s_form
from bonndit.utilc.lowrank cimport approx_initial
from bonndit.utilc.structures cimport dj_o4
from bonndit.utilc.quaternions cimport quat2rot, quat2ZYZ, basis2quat, quatmul, quat_inv
from bonndit.utilc.watsonfitwrapper cimport *
from .ItoW cimport Trafo
cdef int[:,:] permute_poss = np.array([[0,1,2],[0,2,1], [1,0,2], [1,2,0], [2,1,0], [2,0,1]], dtype=np.int32)
from .kalman.model cimport AbstractModel, fODFModel, MultiTensorModel, BinghamModel, WatsonModel, BinghamQuatModel
from .kalman.kalman cimport Kalman, KalmanQuat
from .alignedDirection cimport Probabilities
from libc.math cimport pow, pi, acos, floor, fabs,fmax, exp, log
from libc.stdio cimport printf
from bonndit.utilc.cython_helpers cimport fa, dctov, sphere2cart, r_z_r_y_r_z
from bonndit.utilc.blas_lapack cimport *
from bonndit.utilc.trilinear cimport trilinear_v, trilinear_v_amb
DTYPE = np.float64
cimport numpy as np
cdef double _lambda_min = 0.01
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

###
# Given the ItoW trafo matrix and a cuboid of 8 points shape [2, 2, 2, 3] with a vector for each edge compute the
# trilinear
# interpolation

cdef int[:] permutation = np.array([0,1,2]*8, dtype=np.int32)
cdef double[:,:] neigh = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=DTYPE)
cdef int[:] minus = np.zeros((3,), dtype=np.int32)
cdef int[:,:] neigh_int = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)], dtype=np.int32)
cdef int[:] best = np.zeros((4*8,), dtype=np.int32),  old_best = np.zeros((8,), dtype=np.int32)
cdef double[:,:,:] test_cuboid = np.zeros((8, 3, 3), dtype=DTYPE)
cdef double[:] placeholder = np.zeros((3,), dtype=DTYPE)


cdef class Interpolation:

    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
        self.vector_field = vector_field
        self.vector = np.zeros((3,), dtype=DTYPE)
        self.cuboid = np.zeros((8, 3, 3), dtype=DTYPE)
        self.floor_point = np.zeros((8, 3), dtype=DTYPE)
        self.inv_trafo = np.linalg.inv(kwargs['trafo_data'])
        self.point_index = np.zeros((4,), dtype=DTYPE)
        self.point_world = np.zeros((4,), dtype=DTYPE)
        self.best_dir  = np.zeros((3,3), dtype=DTYPE)
        self.next_dir = np.zeros((3,), dtype=DTYPE)
        self.cache = np.zeros((grid[0], grid[1], grid[2], 4 * 8), dtype=np.int32)
        self.best_ind = 0
        self.prob = probClass
        self.loss = 0
        self.u = 0
        if vector_field.shape[0] == 5:
            u = 1

    cdef bint check_point(self, double[:] point):
       # print(np.array(self.vector_field.shape), np.array(point))
        if self.vector_field.shape[2] > point[0] > 0 and self.vector_field.shape[3] > point[1] > 0 and self.vector_field.shape[4] > point[2] > 0:
            return True
        return False





    cdef void calc_cube(self,double[:] point) : # : # : # nogil:
        """ This function calculates the cube around a point.

        Parameters
        ----------
        point 3 dimensional point


        Returns
        -------

        """
        add_pointwise(self.floor_point, neigh, point)
        floor_pointwise_matrix(self.floor_point, self.floor_point)

    cdef void nearest_neigh(self,double[:] point) : # : # : # nogil:
        """ Return the nearest neighbour to a given point by l2 norm. Therefore uses the cube around a point and calc
        the distance to all other points.

        Parameters
        ----------
        point

        Returns
        -------

        """
        cdef double dist, act_dist
        cdef int index, best_ind
        self.calc_cube(point)
        # since linearity this works. Otherwise we first have to shift and the calc distances. For linear functions
        # it doesnt matter.
        sub_vectors(self.vector, self.floor_point[0], point)
        dist = norm(self.vector)
        # find nearest neighbor
        for index in range(8):
            sub_vectors(self.vector, self.floor_point[index], point)
            act_dist = norm(self.vector)
            if act_dist <= dist:
                dist = act_dist
                best_ind = index
        self.best_ind = best_ind

    #somehow slicing is not possible
    cdef void set_vector(self, int index, int j) : # : # : # nogil:
        cdef int i
        for i in range(3):
            self.vector[i] = self.vector_field[i + 1 + self.u, j, int(self.floor_point[index, 0]),
                          int(self.floor_point[index, 1]), int(self.floor_point[index, 2])]


    #@Cython.cdivision(True)
    cdef bint main_dir(self, double[:] point) : # : # : # nogil:
            cdef double zero = 0
            self.point_world[:3] = point
            self.point_world[3] = 1
            cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 4, 1, &self.inv_trafo[0, 0], 4, &self.point_world[0], 1, 0,
                        &self.point_index[0], 1)


            if self.check_point(self.point_index[:3]):
                self.nearest_neigh(self.point_index[:3])
                self.set_vector(self.best_ind, 0)
                mult_with_scalar(self.next_dir, pow(self.vector_field[0, 0, int(self.floor_point[                                                                                                       self.best_ind, 0]),
                                                                   int(self.floor_point[ self.best_ind, 1]),
                                                                  int(self.floor_point[self.best_ind, 2])], 0.25), self.vector)
                return True
            return False



    cpdef int interpolate(self,double[:] point, double[:] old_dir, int r) except *: # : # : # nogil except *:
        pass


cdef class FACT(Interpolation):
    cpdef int interpolate(self, double[:] point, double[:] old_dir, int r) except *: # : # : # nogil except *:
        cdef int i
        cdef double l, max_value
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)

        self.nearest_neigh(self.point_index[:3])
        max_value = fmax(fmax(self.vector_field[0, 0, int(self.floor_point[self.best_ind, 0]),
                                                int(self.floor_point[self.best_ind,1]),
                                           int(self.floor_point[self.best_ind, 2])],
                                            self.vector_field[0, 1,int(self.floor_point[self.best_ind, 0]),
                                                              int(self.floor_point[self.best_ind,1]),
                                                              int(self.floor_point[self.best_ind, 2])]),
                         self.vector_field[0, 2, int(self.floor_point[self.best_ind, 0]),
                                           int(self.floor_point[self.best_ind,1]),int(self.floor_point[self.best_ind, 2])])

        for i in range(3):
            if self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]), int(self.floor_point[self.best_ind,
                                                                                                     1]),
                                 int(self.floor_point[self.best_ind, 2])] > max_value/10:
                l = pow(fabs(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]),
                        int(self.floor_point[self.best_ind, 1]),int(self.floor_point[self.best_ind,
                                                                    2])]), 1/4)
            else:
                l = 0
            self.set_vector(self.best_ind, i)
            mult_with_scalar(self.best_dir[i], l, self.vector)
        #printf('%i \n', thread_id)<
        self.prob.calculate_probabilities(self.best_dir, old_dir)
        mult_with_scalar(self.next_dir, 1, self.prob.best_fit)
        return 0


cdef double[:] valsec = np.empty([1, ], dtype=DTYPE), \
                val= np.empty([1,], dtype=DTYPE), \
                testv = np.empty([3,], dtype=DTYPE), \
                anisoten = np.empty([15,],  dtype=DTYPE), \
                isoten = np.empty([15,],dtype=DTYPE), \
                der = np.empty([3,],  dtype=DTYPE)

cdef double[:, :] tens = np.zeros([3, 15], dtype=DTYPE)

cdef class TrilinearFODF(Interpolation):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
        super(TrilinearFODF, self).__init__(vector_field, grid, probClass, **kwargs)
        self.data = kwargs['data']
        self.fodf = np.zeros((16,))
        self.fodf1 = np.zeros((16,))
        self.length = np.zeros((3,))
        self.inc= np.int32(np.prod(kwargs['data'].shape[1:]))
        self.empty = np.zeros((15,))
        if kwargs['r'] == -1:
            kwargs['r'] = 2
            while True:
                kwargs['r'] += 1
                if len(np.array(
                    [[i, j, k] for i in range(-int(kwargs['r']), int(kwargs['r']) + 1) for j in range(-int(kwargs['r']), int(kwargs['r']) + 1) for k in
                     range(-int(kwargs['r']), int(kwargs['r']) + 1) if np.linalg.norm(np.dot(kwargs['trafo'], np.array([i, j, k]))) <= kwargs['r']],
                    dtype=np.intc)) >= 90:
                    break
            self.auto = True
            self.r = kwargs['r']
        else:
            self.auto = False
            self.r = kwargs['r']


        if kwargs['sigma_2'] == 0:
            self.sigma_2 = ((np.linalg.norm(kwargs['trafo'] @ np.array((1, 0, 0))) + \
                             np.linalg.norm(kwargs['trafo'] @ np.array((0, 1, 0))) + np.linalg.norm(kwargs['trafo'] @ np.array((0, 0, 1)))) / 3)
        else:
            self.sigma_2 = kwargs['sigma_2']
        self.neighbors = np.array(sorted([[i, j, k] for i in range(-int(kwargs['r']), 1 + int(kwargs['r'])) for j in
                                          range(-int(kwargs['r']), 1 + int(kwargs['r'])) for k in
                                          range(-int(kwargs['r']), 1 + int(kwargs['r']))],
                                         key=lambda x: np.linalg.norm(kwargs['trafo'] @ x)), dtype=np.int32)
        if kwargs['sigma_1'] == 0:
            skip = np.zeros(kwargs['data'].shape[1:])
            neighbors = np.array([x for x in self.neighbors if np.linalg.norm(kwargs['trafo'] @ x) <= kwargs['r']])
            var = np.zeros((len(neighbors), ) + kwargs['data'].shape[1:])

            for i,j,k in tqdm(np.ndindex(kwargs['data'].shape[1:]), total=np.prod(kwargs['data'].shape[1:])):
                if kwargs['data'][0, i,j,k] == 0.0:
                    skip[i,j,k] = 1
                    continue

                for index in range(neighbors.shape[0]):
                    if kwargs['data'].shape[1] > i + neighbors[index, 0] >= 0 \
                        and kwargs['data'].shape[2] > j + neighbors[index, 1] >= 0 \
                        and kwargs['data'].shape[3] > k + neighbors[index, 2] >= 0:
                        sub_vectors(isoten, kwargs['data'][1:, i,j,k], kwargs['data'][1:, i + neighbors[index, 0], j + neighbors[index, 1], k + neighbors[index, 2]])
                        var[index, i,j,k] = hota_4o3d_sym_norm(isoten)
                    else:
                        skip[i,j,k] = 1
            #nu = np.mean(var[:, skip == 0])
            self.sigma_1 = np.median(var[:, skip == 0])**2
        else:
            self.sigma_1 = kwargs['sigma_1']
        self.best_dir_approx = np.zeros((3,3))

        self.point_diff = np.zeros((3,), dtype=DTYPE)
        self.vlinear = np.zeros((8, kwargs['data'].shape[0]))
        self.trafo = kwargs['trafo']
        self.dist = np.zeros((3,), dtype=DTYPE)
        self.r = kwargs['r']
        self.rank = kwargs['rank']
        print(self.sigma_2, self.sigma_1, self.r)


    cdef void trilinear(self, double[:] point) : # : # : # nogil except *:
        cdef int i, j, k, m,n,o
        for i in range(8):
            j = <int> floor(i / 2) % 2
            k = <int> floor(i / 4) % 2
            m = <int> point[0] + i%2
            n = <int> point[1] + j
            o = <int> point[2] + k
            #check for out of bounds:
            if m<0 or n<0 or o<0 or m>=self.data.shape[1] or n>=self.data.shape[2] or o>=self.data.shape[3]:
                cblas_dscal(self.vlinear.shape[1], 0, &self.vlinear[i,0], 1)
            else:
                dm2toc(&self.vlinear[i, 0], self.data[:, m,n,o],  self.vlinear.shape[1])

        for i in range(4):
            cblas_dscal(self.vlinear.shape[1], (1 + floor(point[2]) - point[2]), &self.vlinear[i, 0], 1)
            cblas_daxpy(self.vlinear.shape[1], (point[2] - floor(point[2])), &self.vlinear[4+i, 0], 1, &self.vlinear[i,0], 1)
        for i in range(2):
            cblas_dscal(self.vlinear.shape[1], (1 + floor(point[1]) - point[1]), &self.vlinear[i, 0], 1)
            cblas_daxpy(self.vlinear.shape[1], (point[1] - floor(point[1])), &self.vlinear[2 + i, 0], 1, &self.vlinear[i, 0], 1)
        cblas_dscal(self.vlinear.shape[1], (1 + floor(point[0]) - point[0]), &self.vlinear[0, 0], 1)
        cblas_daxpy(self.vlinear.shape[1], (point[0] - floor(point[0])), &self.vlinear[1,0], 1, &self.vlinear[0,0], 1)
        cblas_dcopy(self.vlinear.shape[1], &self.vlinear[0,0], 1, &self.fodf[0], 1)

    cdef void neigh(self, double[:] point) : # : # : # nogil except *:
        cdef double x, scale = 0, dis=0, distance =0
        cdef int i, index, p_0 = <int> point[0], p_1 = <int> point[1], p_2 = <int> point[2], pw_0, pw_1, pw_2
        self.trilinear(point)
        cblas_dscal(16,0, &self.fodf1[0],1)
        for index in range(<int> self.neighbors.shape[0]):
            pw_0 = p_0 + self.neighbors[index, 0]
            pw_1 = p_1 + self.neighbors[index, 1]
            pw_2 = p_2 + self.neighbors[index, 2]
            for i in range(3):
                self.point_diff[i] = self.neighbors[index, i] - point[i]%1
            cblas_dgemv(CblasRowMajor, CblasNoTrans, 3,3,1, &self.trafo[0,0], 3, &self.point_diff[0], 1, 0, &self.dist[0], 1)
            distance = cblas_dnrm2(3, &self.dist[0], 1)
            if distance > self.r or (index>27 and self.auto):
                break
            if self.data.shape[1] > pw_0 >= 0 and self.data.shape[2] > pw_1 >= 0 and self.data.shape[3] > pw_2 >= 0:
                sub_vectors(self.empty, self.fodf[1:], self.data[1:, pw_0, pw_1, pw_2])
                x = hota_4o3d_sym_norm(self.empty)
                dis = exp(-(x*x)/self.sigma_1 - distance**2/self.sigma_2)
                scale += dis
                for i in range(16):
                    self.fodf1[i] += dis*self.data[i, pw_0, pw_1, pw_2]
                #cblas_daxpy(16, dis, &self.data[0, pw_0, pw_1, pw_2], 1, &self.fodf1[0], 1)
        if scale > 0:
            mult_with_scalar(self.fodf, 1/scale, self.fodf1)

    cpdef int interpolate(self, double[:] point, double[:] old_dir, int r) except *: # : # : # nogil except *:
    #   with gil: print(np.array(old_dir))
        # Initialize with last step. Except we are starting again.
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
        if r==0:
            cblas_dscal(9,0, &self.best_dir[0,0],1)
            cblas_dscal(3,0, &self.length[0],1)
        # If self.r==0: Interpolate trilinear else: calculate average over neighborhood.

        cdef int i
        if self.r==0:
            self.trilinear(self.point_index[:3])
        else:
            self.neigh(self.point_index[:3])
        if self.fodf[0] == 0:
            return -1
        #set_zero_matrix(tens)
        for i in range(3):
            hota_4o3d_sym_eval(tens[i, :], self.length[i], self.best_dir_approx[:, i])
            cblas_daxpy(15, -1, &tens[i,0], 1, &self.fodf[1], 1)
#       tijk_approx_rankk_3d_f(*self.length[0], *self.best_dir_approx[0,0], )
        approx_initial(self.length, self.best_dir_approx, tens, self.fodf[1:], self.rank, valsec, val,der, testv, anisoten, isoten)

        for i in range(3):
            if self.length[i] > 0.1:
                mult_with_scalar(self.best_dir[i], pow(self.length[i], 1/4), self.best_dir_approx[:,i])
            else:
                mult_with_scalar(self.best_dir[i], 0, self.best_dir_approx[:,i])
        self.prob.calculate_probabilities(self.best_dir, old_dir)
        cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0],1)
        return 0




cdef class Trilinear(Interpolation):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
        super(Trilinear, self).__init__(vector_field, grid, probClass, **kwargs)
        self.array = np.zeros((2,3), dtype=DTYPE)
        self.x_array = np.zeros((4,3), dtype=DTYPE)
        self.point = np.zeros((3,), dtype=DTYPE)
        self.dir = np.zeros((8, 3, 3), dtype=DTYPE)
        self.new_best_dir = np.zeros((3, 3), dtype=DTYPE)
        #self.cache = np.zeros((grid[0], grid[1], grid[2], 4*8), dtype=np.int32)
        self.permutation = np.zeros((16,), dtype=np.int32)
        self.not_check = np.zeros((3,2), dtype=np.int32)
        self.floor = np.zeros((3,), dtype=np.int32)



    cdef void set_array(self, int array, int index, int i) : # : # : # nogil:
        self.set_vector(index, i)
        if self.vector_field[0, i, int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),
                             int(self.floor_point[index, 2])] != 0 and self.vector_field[0, i, int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),
                             int(self.floor_point[index, 2])] == self.vector_field[0, i, int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),
                             int(self.floor_point[index, 2])]:
            mult_with_scalar(self.array[array], pow(self.vector_field[0, i, int(self.floor_point[index, 0]),
                                                                      int(self.floor_point[index, 1]),
                                                                  int(self.floor_point[index, 2])], 1 / 4),
                             self.vector)
        else:
            mult_with_scalar(self.array[array], 0, self.vector)



    cpdef best_dirp(self):
        return np.array(self.best_dir)

    cpdef set_best_dirp(self, double[:,:] best_dir):
        self.best_dir = best_dir

    cpdef get_cache(self):
        return np.array(self.cache)

    cpdef get_next_dir(self):
        return np.array(self.next_dir[0:3])

    cpdef set_cache(self, int[:,:,:,:] cache):
        self.cache = cache



    cpdef int interpolate(self, double[:] point, double[:] old_dir, int r) except *: # : # : # nogil except *:
        """ This function calculates the interpolation based on https://en.wikipedia.org/wiki/Trilinear_interpolation
        for each vectorfield. Afterwards the we chose randomly from the 3 vectors.

        Parameters
        ----------
        point   Point in plane
        old_dir direction in point

        Returns
        -------

        """
        cdef int i, j
        cdef int con = 1
        cdef double test=0
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
        self.calc_cube(self.point_index[:3])
        for i in range(3):
            self.floor[i] = int(self.point_index[i])
        # Check if the best dir is initialized. If no initizialize first with the nearest neighbor. Then fit neighbors.
        for i in range(3):
            test+=norm(self.best_dir[i])
        if test==0:
            self.nearest_neigh(self.point_index[:3])
            for i in range(3):
                if point_validator(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]),
                                                     int(self.floor_point[self.best_ind,1]),int(self.floor_point[self.best_ind, 2])], 1):
                    l = pow(fabs(self.vector_field[0, i, int(self.floor_point[self.best_ind, 0]),
                                                   int(self.floor_point[self.best_ind, 1]), int(self.floor_point[self.best_ind,2])]), 1 / 4)
                else:
                    l = 0
                self.set_vector(self.best_ind, i)
                mult_with_scalar(self.best_dir[i], l, self.vector)

        if sum_c_int(self.cache[self.floor[0], self.floor[1], self.floor[2],:]) != 0:
            #print('hier wird permutiert')
            self.permute(self.point_index[:3])
        else:
            con = self.kmeans(self.point_index[:3])
            self.permute(self.point_index[:3])
        #else:
        #   self.permute(point)
        if con:

            for i in range(3):
                self.point[i] = self.point_index[i]%1

            for i in range(3):
                # interpolate in x direction
                for j in range(4):
                    mult_with_scalar(self.array[0], self.point[0], self.cuboid[4+j, i])
                    mult_with_scalar(self.array[1], 1-self.point[0], self.cuboid[j, i])
                    add_vectors(self.x_array[j], self.array[0], self.array[1])

                # then in y direction
                mult_with_scalar(self.x_array[0], 1-self.point[1], self.x_array[0])
                mult_with_scalar(self.x_array[2], self.point[1], self.x_array[2])
                add_vectors(self.array[0], self.x_array[0], self.x_array[2])
                mult_with_scalar(self.x_array[1], 1-self.point[1], self.x_array[1])
                mult_with_scalar(self.x_array[3], self.point[1], self.x_array[3])
                add_vectors(self.array[1], self.x_array[1], self.x_array[3])

                # then z direction
                mult_with_scalar(self.array[0], 1-self.point[2], self.array[0])
                mult_with_scalar(self.array[1], self.point[2], self.array[1])
                add_vectors(self.best_dir[i], self.array[0], self.array[1])


            self.prob.calculate_probabilities(self.best_dir, old_dir)
            self.next_dir = self.prob.best_fit

        else:
            mult_with_scalar(self.next_dir, 0, self.prob.best_fit)
        #TODO change back
        return con

#   ### TODO is here a better way?
    cdef void permute(self, double[:] point) : # : # : # nogil except *:
        """ Little anoying... If a cube was already cached, this function uses the cache to set the dir parameter
        accordingly. We loop through all indices and for each index through the 2 saved vectors. The two saved
        vectors are set to the first two directions and the remaining is set to the last direction. To get the last
        one it should be enough to calculate 3-a-b where a and b are the indices of the other two directions.

        Since only the first two are saved

        Parameters
        ----------
        mapping Vector dimension 16 with entries from 0 to two which define the permutation

        Returns
        -------

        """
        cdef int i, index, k, l, z = 1
        cdef double exponent = 0
        for index in range(8):
            for i in range(3):
                if point_validator(self.vector_field[0, permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]), int(self.floor_point[index, 1]), int(self.floor_point[index, 2])], 1):
                    exponent = pow(fabs(self.vector_field[0,  permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]), int(self.floor_point[index, 1]),int(self.floor_point[index, 2])]), 1/4)
                else:
                    exponent = 0
                for k in range(3):
                    placeholder[k] = self.vector_field[1 + k, permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),int(self.floor_point[index, 2])]
                for k in range(3):
                    self.cuboid[index,i,k] = exponent * self.cache[int(point[0]), int(point[1]),int(point[2]), index*4 + i + 1] * self.vector_field[1 +k, permute_poss[self.cache[int(point[0]), int(point[1]),int(point[2]), index*4], i], int(self.floor_point[index, 0]),int(self.floor_point[index, 1]),int(self.floor_point[index, 2])]


    cdef void set_new_poss(self) : # : # : # nogil except *:
        cdef int i,j,k
        for i in range(8):
            set_zero_matrix(self.cuboid[i])
        for i in range(8):
            for j in range(3):
                for k in range(3):
                    self.cuboid[i, j, k] = best[4*i + j + 1] * test_cuboid[i, permute_poss[best[4*i], j], k]

    cdef int kmeans(self, double[:] point) : # : # : # nogil except *:
        cdef int i, j, k, l, max_try=0, best_min=0
        cdef double exponent = 0, best_angle=0, min_angle=0, con=0, test_angle=0
        for i in range(8):
            set_zero_matrix(test_cuboid[i])

        for i in range(8):
            for j in range(3):
                if point_validator(self.vector_field[0, j, int(self.floor_point[i, 0]), int(self.floor_point[i, 1]), int(self.floor_point[i, 2])], 1):
                    exponent = pow(fabs(self.vector_field[0, j, int(self.floor_point[i, 0]),
                                                       int(self.floor_point[i, 1]),
                                                       int(self.floor_point[i, 2])]), 1/4)
                else:
                    exponent = 0

                for k in range(3):
                    test_cuboid[i,j,k] = exponent *  self.vector_field[1 + k, j, int(self.floor_point[i, 0]),int(self.floor_point[i, 1]),int(self.floor_point[i, 2])]
        #for i in range(8):
        # Init best dir. Since it won't work good otherwise....
    #   for j in range(3):
    #       add_vectors(self.best_dir[j], self.best_dir[j], test_cuboid[7, j])
        while True:
            con = 0
            max_try += 1
            # each corner
            for i in range(8):
                min_angle = 0
                for j in range(6):
                    test_angle=0
                    set_zero_vector_int(minus)
                    for k in range(3):
                        ang = angle_deg(self.best_dir[k], test_cuboid[i, permute_poss[j,k]])
                        if ang > 90:
                            mult_with_scalar(placeholder, -1, test_cuboid[i, permute_poss[j, k]])
                            minus[k] = -1
                        else:
                            mult_with_scalar(placeholder, 1, test_cuboid[i, permute_poss[j, k]])
                            minus[k] = 1
                        sub_vectors(placeholder, placeholder, self.best_dir[k])
                        test_angle += norm(placeholder)
                    if min_angle == 0 or test_angle < min_angle:
                        min_angle = test_angle
                    #   with gil: print(test_angle, min_angle)
                        best[4*i] = j
                        for u in range(3):
                            best[4*i + u + 1] = minus[u]

        #   set_zero_matrix(self.best_dir)
        #   for i in range(8):
                    for u in range(3):
                        if norm(self.best_dir[u]) == 0:
                            mult_with_scalar(placeholder, best[4*i+ 1 +u]/8, test_cuboid[i, permute_poss[best[4*i], u]])
                            add_vectors(self.best_dir[u], self.best_dir[u], placeholder)
            for i in range(8):
                con += fabs(best[4*i] - old_best[i])
                old_best[i] = best[4*i]
            if con == 0:
                con = 1
                for i in range(32):
                    self.cache[self.floor[0], self.floor[1], self.floor[2], i] = int(best[i])
                #self.set_new_poss()
                break

            if max_try == 1000:
                con = 0
                #with gil: print('I do not converge')
                break
        set_zero_matrix(self.best_dir)
        return int(con)

cdef class UKF(Interpolation):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
        super(UKF, self).__init__(vector_field, grid, probClass, **kwargs)
        self.mean = np.zeros((kwargs['dim_model'],), dtype=np.float64)
        self.mlinear  = np.zeros((8,kwargs['data'].shape[3]), dtype=np.float64) ##  Shpuld be always 8. For edges of cube.
        self.P = np.zeros((kwargs['dim_model'],kwargs['dim_model']), dtype=np.float64)
        self.y = np.zeros((kwargs['data'].shape[3],), dtype=np.float64)
        if kwargs['baseline'] != "" and kwargs["model"] == "MultiTensor":
            self.data = kwargs['data']/kwargs['baseline'][:,:,:,np.newaxis]
        else:
            self.data = kwargs['data']
        if kwargs['model'] == 'LowRank':
            self._model = fODFModel(vector_field=vector_field, **kwargs)
        elif kwargs['model'] == 'Watson':
            self._model = WatsonModel(vector_field=vector_field, **kwargs)
        elif kwargs['model'] == 'Bingham':
            self._model = BinghamModel(vector_field=vector_field, **kwargs)
        else:
            self._model = MultiTensorModel(**kwargs)
        self._kalman = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model)

    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *: # : # : # nogil except *:
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
        cdef int z, i, info = 0
        # Interpolate current point
        trilinear_v(self.point_index[:3], self.y, self.mlinear, self.data)
        # If we are at the seed. Initialize the Kalmanfilter
        if restart == 0:
            #print("Restart \n")
            self._model.kinit(self.mean, self.point_index[:3], old_dir, self.P, self.y)
        # Run Kalmannfilter
        info = self._kalman.update_kalman_parameters(self.mean, self.P, self.y)
        return self.select_next_dir(info, old_dir)

    cdef int select_next_dir(self, int info, double[:] old_dir):
        return info

cdef class UKFFodfAlt(Interpolation):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
        dim_model = kwargs['dim_model']
        kwargs['dim_model'] = 4
        self.num_kalman = dim_model//4
        self.mean = np.zeros((dim_model//4,4), dtype=np.float64)
        self.mlinear  = np.zeros((8,kwargs['data'].shape[3]), dtype=np.float64) ##  Shpuld be always 8. For edges of cube.
        self.P = np.zeros((dim_model//4, 4, 4), dtype=np.float64)
        self.y = np.zeros((dim_model//4, kwargs['data'].shape[3],), dtype=np.float64)
        self.res = np.zeros((kwargs['data'].shape[0],), dtype=np.float64)
        self._model1 = fODFModel(vector_field=vector_field[:, 0:1], **kwargs)
        self._kalman1 = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model1)
        self._model2 = fODFModel(vector_field=vector_field[:, 1:2], **kwargs)
        self._kalman2 = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model2)
        self.data = kwargs['data']
        self.prob = prob

    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *: # : # : # nogil except *:
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
        cdef int i, info = 0
        # Interpolate current point

        trilinear_v(self.point_index[:3], self.y[0], self.mlinear, self.data)
        trilinear_v(self.point_index[:3], self.y[1], self.mlinear, self.data)
        # If we are at the seed. Initialize the Kalmanfilter
        if restart == 0:
            #with gil:
            self._model1.kinit(self.mean[0], self.point_index[:3], old_dir, self.P[0], self.y[0])
            self._model2.kinit(self.mean[1], self.point_index[:3], old_dir, self.P[1], self.y[1])

        # Run Kalmannfilter
        for i in range(self.num_kalman):
            for j in range(self.num_kalman):
                if i == j:
                    continue
                if self._model1.order ==4:
                    hota_4o3d_sym_eval(self.res, self.mean[j, 3], self.mean[j, :3])
                else:
                    hota_6o3d_sym_eval(self.res, self.mean[j, 3], self.mean[j, :3])
                cblas_daxpy(self.res.shape[0], -1, &self.res[0], 1, &self.y[i,0], 1)
            if i == 0:
                self._kalman1.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])
            else:
                self._kalman2.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])


        for i in range(self.num_kalman):
            if cblas_dnrm2(3, &self.mean[i,0], 1) != 0:
                cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[i,0], 1), &self.mean[i,0], 1)
                if cblas_ddot(3, &self.mean[i,0], 1, &old_dir[0],1) < 0:
                    cblas_dscal(3, -1, &self.mean[i,0], 1)
                self.mean[i, 3] = max(self.mean[i,3],_lambda_min)
            else:
                cblas_dscal(3, cblas_dnrm2(3, &self.mean[i,0], 1), &self.mean[i,0], 1)
                self.mean[i, 3] = max(self.mean[i,3], _lambda_min)

        if True:
            trilinear_v(self.point_index[:3], self.y[0], self.mlinear, self.data)
            for i in range(2):
                hota_4o3d_sym_eval(self.res, self.mean[i, 3], self.mean[i, :3])
                cblas_daxpy(self.res.shape[0], -1, &self.res[0], 1, &self.y[0, 0], 1)
            #print(np.linalg.norm(self.y[0]))
##

        for i in range(self._model1.num_tensors):
            dctov(&self.mean[i,0], self.best_dir[i])
            if self.mean[i,3] > 0.1:
                #print(self.mean[4 * i + 3])
                cblas_dscal(3,pow(self.mean[i, 3], 0.25), &self.best_dir[i,0],1)
            else:
                cblas_dscal(3,0, &self.best_dir[i,0],1)

        self.prob.calculate_probabilities(self.best_dir, old_dir)
        self.next_dir = self.prob.best_fit
        return 0

cdef class UKFFodf(UKF):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
        super(UKFFodf, self).__init__(vector_field, grid, prob, **kwargs)

    cdef int select_next_dir(self, int info, double[:] old_dir):
        if info != 0:
            return info


        for i in range(self._model.num_tensors):
            if cblas_dnrm2(3, &self.mean[4*i], 1) != 0:
                cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[4*i], 1), &self.mean[4*i], 1)
                if cblas_ddot(3, &self.mean[4*i], 1, &old_dir[0],1) < 0:
                    cblas_dscal(3, -1, &self.mean[4*i], 1)
                self.mean[4*i+3] = max(self.mean[4*i+3],_lambda_min)
            else:
                cblas_dscal(3, cblas_dnrm2(3, &self.mean[4 * i], 1), &self.mean[4 * i], 1)
                self.mean[4 * i + 3] = max(self.mean[4 * i + 3], _lambda_min)


        for i in range(self._model.num_tensors):
            dctov(&self.mean[4*i], self.best_dir[i])
            if self.mean[4 * i + 3] > 0.1:
                #print(self.mean[4 * i + 3])
                cblas_dscal(3,pow(self.mean[4 * i + 3], 0.25), &self.best_dir[i,0],1)
            else:
                cblas_dscal(3,0, &self.best_dir[i,0],1)

        self.prob.calculate_probabilities(self.best_dir, old_dir)
        self.next_dir = self.prob.best_fit

        return info


cdef class UKFWatson(UKF):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
        super(UKFWatson, self).__init__(vector_field, grid, prob, **kwargs)
        self.kappas = np.zeros(<int> (kwargs['dim_model'] // 5), dtype=DTYPE)
        self.weights = np.zeros(<int> (kwargs['dim_model'] // 5), dtype=DTYPE)
        self._model1 = WatsonModel(vector_field=vector_field, **kwargs)
        self.store_loss = kwargs['store_loss']


    cdef int select_next_dir(self, int info, double[:] old_dir):
        if info != 0:
            return info


        for i in range(self._model.num_tensors):
            if cblas_dnrm2(3, &self.mean[5*i + 2], 1) != 0:
                cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[5*i  + 2], 1), &self.mean[5*i + 2], 1)
            else:
                cblas_dscal(3, cblas_dnrm2(3, &self.mean[5 * i+2], 1), &self.mean[5 * i+2], 1)
            self.mean[5 * i + 1] = max(self.mean[5 * i + 1], _lambda_min)
            self.mean[5 * i] = min(max(self.mean[5 * i], _lambda_min), log(80))
            self.weights[i] = fabs(self.mean[i * 5 + 1])
            self.kappas[i] = exp(self.mean[i * 5])
            cblas_dcopy(3, &self.mean[i*5+2], 1,  &self.best_dir[i, 0], 1)


        if self.store_loss:
            trilinear_v(self.point_index[:3], self.y, self.mlinear, self.data)
            for i in range(self._model.num_tensors):
                cblas_dscal(self._model1.dipy_v.shape[0], 0, &self._model1.dipy_v[0], 1)
                c_sh_watson_coeffs(self.kappas[i], &self._model1.dipy_v[0], self._model1.order)
                self.mean[i*5+2] *= -1
                self.mean[i*5+4] *= -1
                cart2sphere(self._model1.angles[1:], self.mean[i*5+2:(i+1)*5])
                self.mean[i*5+2] *= -1
                self.mean[i*5+4] *= -1
                div = self._model1.sh_norm(self._model1.dipy_v)
                self._model1.dipy_v[0] *= self._model1.rank_1_rh_o4[0]/div
                self._model1.dipy_v[3] *= self._model1.rank_1_rh_o4[1]/div
                self._model1.dipy_v[10] *= self._model1.rank_1_rh_o4[2]/div
                c_map_dipy_to_pysh_o4(&self._model1.dipy_v[0], &self._model1.pysh_v[0])
                c_sh_rotate_real_coef(&self._model1.rot_pysh_v[0], &self._model1.pysh_v[0], self._model1.order, &self._model1.angles[0], &dj_o4[0][0][0])
                c_map_pysh_to_dipy_o4(&self._model1.rot_pysh_v[0],&self._model1.dipy_v[0])
                cblas_daxpy(self.y.shape[0], -self.weights[i], &self._model1.dipy_v[0], 1, &self.y[0], 1)
            self.loss = cblas_dnrm2(self.y.shape[0], &self.y[0], 1)
        #print(self.loss)
        self.prob.calculate_probabilities_sampled(self.best_dir, self.kappas, self.weights, old_dir, self.point_index[:3])
        cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0], 1)
        return info


    ## TODO Das überarbeiten:
cdef class UKFWatsonAlt(Interpolation):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
        self.kappas = np.zeros((kwargs['dim_model'] // 5, ), dtype=DTYPE)
        self.weights = np.zeros((kwargs['dim_model'] // 5, ), dtype=DTYPE)
        dim_model = kwargs['dim_model']
        kwargs['dim_model'] = 5
        self.num_kalman = dim_model//5
        self.mean = np.zeros((dim_model//5,5), dtype=np.float64)
        self.mlinear  = np.zeros((8,kwargs['data'].shape[3]), dtype=np.float64) ##  Shpuld be always 8. For edges of cube.
        self.P = np.zeros((dim_model//5, 5, 5), dtype=np.float64)
        self.y = np.zeros((dim_model//5, kwargs['data'].shape[3]), dtype=DTYPE)
        self.res = np.zeros((kwargs['data'].shape[3]), dtype=np.float64)
        self.res_copy = np.zeros((kwargs['data'].shape[3]), dtype=np.float64)
        self._model1 = WatsonModel(vector_field=vector_field[:, 0:1], **kwargs)
        self._kalman1 = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model1)
        self._model2 = WatsonModel(vector_field=vector_field[:, 1:2], **kwargs)
        self._kalman2 = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model2)
        self.data = kwargs['data']
        self.rot_pysh_v = np.zeros((2 *  5 * 5,), dtype=DTYPE)
        self.pysh_v = np.zeros((2 *  5 * 5,), dtype=DTYPE)
        self.angles  = np.zeros((3,), dtype=DTYPE)
        self.store_loss = kwargs['store_loss']

        self._model = WatsonModel(vector_field=vector_field, **kwargs)

    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *:
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
        cdef int i,j, info = 0
        # Interpolate current point

        trilinear_v(self.point_index[:3], self.y[0], self.mlinear, self.data)
        trilinear_v(self.point_index[:3], self.y[1], self.mlinear, self.data)
        # If we are at the seed. Initialize the Kalmanfilter
        if restart == 0:

            self._model1.kinit(self.mean[0], self.point_index[:3], old_dir, self.P[0], self.y[0])
            self._model2.kinit(self.mean[1], self.point_index[:3], old_dir, self.P[1], self.y[1])

        # Run Kalmannfilter
        for i in range(self.num_kalman):
            for j in range(self.num_kalman):
                if i == j:
                    continue
#
                cblas_dscal(self.res.shape[0], 0, &self.res_copy[0], 1)
                c_sh_watson_coeffs(exp(self.mean[j,0]), &self.res_copy[0], self._model1.order)
                self.res_copy[0] *= self._model1.rank_1_rh_o4[0]
                self.res_copy[3] *= self._model1.rank_1_rh_o4[1]
                self.res_copy[10] *= self._model1.rank_1_rh_o4[2]
                if self._model1.order == 6:
                    self.res_copy[21] *= self._model1.rank_1_rh_o4[3]
                self.mean[j,2] *= -1
                self.mean[j,4] *= -1
                cart2sphere(self.angles[1:], self.mean[j, 2:])
                self.mean[j,2] *= -1
                self.mean[j,4] *= -1
                ## TODOc_
                c_sh_rotate_real_coef_fast(&self.res[0], 1, &self.res_copy[0], 1, self._model.order, &self.angles[0])
                cblas_daxpy(self.res.shape[0], -max(self.mean[j, 1], _lambda_min), &self.res[0], 1, &self.y[i,0], 1)
            if i == 0:
                self._kalman1.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])
            else:
                self._kalman2.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])

        for i in range(self.num_kalman):
            if cblas_dnrm2(3, &self.mean[i, 2], 1) != 0:
                cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[i, 2], 1), &self.mean[i, 2], 1)

            else:
                cblas_dscal(3, cblas_dnrm2(3, &self.mean[i, 2], 1), &self.mean[i, 2], 1)
            self.mean[i, 1] = max(self.mean[i, 1], _lambda_min)
            self.weights[i] = fabs(self.mean[i, 1])
            self.kappas[i] = exp(self.mean[i, 0])
            cblas_dcopy(3, &self.mean[i, 2], 1, &self.best_dir[i,0], 1)


        self.prob.calculate_probabilities_sampled(self.best_dir, self.kappas, self.weights, old_dir, self.point_index[:3])
        cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0], 1)



        return info

cdef class UKFBingham(UKF):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
        super(UKFBingham, self).__init__(vector_field, grid, prob, **kwargs)
        self.A = np.zeros((self._model.num_tensors, 3, 3))
        self.mu = np.zeros((self._model.num_tensors, 3))
        self.l_k_b = np.zeros((self._model.num_tensors, 3))
        self.R = np.zeros((3,3), dtype=DTYPE)
        self._model1 = BinghamModel(vector_field=vector_field, **kwargs)
        self.store_loss = kwargs['store_loss']

    cdef int select_next_dir(self, int info, double[:] old_dir):
        if info != 0:
            return info
        #print(np.array(self.mean))
        for i in range(self._model.num_tensors):
            self.mean[6*i + 0] = max(self.mean[6 * i + 0], _lambda_min)
            self.mean[6*i + 1] = min(max(self.mean[6 * i + 1], log(0.2)), log(50))
            self.mean[6*i + 2] = min(max(self.mean[6 * i + 2], log(0.1)), self.mean[6 * i + 1])
            r_z_r_y_r_z(self.R, self.mean[6 * i+3:6*(i+1)])
            cblas_dscal(3, -1, &self.R[0,0], 3)
            cblas_dscal(3, -1, &self.R[0,2], 3)
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 1, 1, &self.R[0,0], 3, &self.R[0,0], 3, 0, &self.A[i, 0,0], 3)
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 1, -1, &self.R[1,0], 3, &self.R[1,0], 3, 1, &self.A[i, 0,0], 3)
            cblas_dscal(9, exp(self.mean[6 * i + 2]), &self.A[i, 0,0], 1)
            # enough to reorient just mu
            cblas_dcopy(3, &self.R[2,0], 1, &self.mu[i,0], 1)

            self.l_k_b[i, 0] = self.mean[6 * i + 0]
            self.l_k_b[i, 1] = exp(self.mean[6 * i + 1])
            self.l_k_b[i, 2] = exp(self.mean[6 * i + 2])

        if self.store_loss:
            trilinear_v(self.point_index[:3], self.y, self.mlinear, self.data)
            base = cblas_dnrm2(self.y.shape[0], &self.y[0], 1)
            for i in range(self._model.num_tensors):
                kappa = exp(self.mean[i*6 + 1])
                beta = exp(self.mean[i*6 + 2])
                self._model1.sh_bingham_coeffs(kappa, beta)
                cblas_dcopy(3, &self.mean[i*6 + 3], 1, &self._model1.angles[0], 1)

                c_sh_rotate_real_coef_fast(&self._model1.dipy_v[0], 1, &self._model1.dipy_v[0], 1, 4, &self._model1.angles[0])
                cblas_daxpy(self.y.shape[0], -self.mean[i*6], &self._model1.dipy_v[0], 1, &self.y[0], 1)
            self.loss = cblas_dnrm2(self.y.shape[0], &self.y[0], 1)

            #print(self.loss/base)
        self.prob.calculate_probabilities_sampled_bingham(self.mu, old_dir, self.A, self.l_k_b)
        cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0], 1)

        return info


cdef class UKFBinghamAlt(Interpolation):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
        self.kappas = np.zeros((kwargs['dim_model'] // 6, ), dtype=DTYPE)
        self.weights = np.zeros((kwargs['dim_model'] // 6, ), dtype=DTYPE)
        dim_model = kwargs['dim_model']
        kwargs['dim_model'] = 6
        self.num_kalman = dim_model//6
        self.mean = np.zeros((dim_model//6,6), dtype=np.float64)
        self.mlinear  = np.zeros((8,kwargs['data'].shape[3]), dtype=np.float64) ##  Shpuld be always 8. For edges of cube.
        self.P = np.zeros((dim_model//6, 6, 6), dtype=np.float64)
        self.y = np.zeros((dim_model//6, kwargs['data'].shape[3]), dtype=DTYPE)
        self.res = np.zeros((kwargs['data'].shape[3]), dtype=np.float64)
        self._model1 = BinghamModel(vector_field=vector_field[:, 0:1], **kwargs)
        self._kalman1 = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model1)
        self._model2 = BinghamModel(vector_field=vector_field[:, 1:2], **kwargs)
        self._kalman2 = Kalman(kwargs['data'].shape[3], kwargs['dim_model'], self._model2)
        self.data = kwargs['data']
        self.rot_pysh_v = np.zeros((2 *  5 * 5,), dtype=DTYPE)
        self.pysh_v = np.zeros((2 *  5 * 5,), dtype=DTYPE)
        self.angles  = np.zeros((3,), dtype=DTYPE)
        self.store_loss = kwargs['store_loss']
        self.A = np.zeros((dim_model//6, 3, 3))
        self.mu = np.zeros((dim_model//6, 3))
        self.l_k_b = np.zeros((dim_model//6, 3))
        self.R = np.zeros((3, 3), dtype=DTYPE)
        self._model = BinghamModel(vector_field=vector_field, **kwargs)

    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *:
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
        cdef int i,j, info = 0
        cdef double base = 0
        cdef double kappa, beta
        # Interpolate current point

        trilinear_v(self.point_index[:3], self.y[0], self.mlinear, self.data)
        trilinear_v(self.point_index[:3], self.y[1], self.mlinear, self.data)
        # If we are at the seed. Initialize the Kalmanfilter
        if restart == 0:
            #with gil:
            self._model1.kinit(self.mean[0], self.point_index[:3], old_dir, self.P[0], self.y[0])
            self._model2.kinit(self.mean[1], self.point_index[:3], old_dir, self.P[1], self.y[1])

        # Run Kalmannfilter
        for i in range(self.num_kalman):
            for j in range(self.num_kalman):
                if i == j:
                    continue
                #print(self.mean[j, 1], self.mean[j, 2])
                kappa = min(max(exp(self.mean[j, 1]), 0.2), 89)
                beta = min(max(exp(self.mean[j, 2]), 0.2), kappa)
                #print(kappa, beta)

                self._model.sh_bingham_coeffs(kappa, beta)

                cblas_dcopy(3, &self.mean[j, 3], 1, &self._model.angles[0], 1)
                c_sh_rotate_real_coef_fast(&self.res[0], 1, &self._model.dipy_v[0], 1, 4, &self.angles[0])
                cblas_daxpy(self.res.shape[0], -max(self.mean[j, 0], _lambda_min), &self.res[0], 1, &self.y[i,0], 1)
            if i == 0:
                self._kalman1.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])
            else:
                self._kalman2.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])
        #print(np.array(self.mean))
        for i in range(self.num_kalman):
            self.mean[i, 0] = max(self.mean[i, 0], _lambda_min)
            self.mean[i, 1] = min(max(self.mean[i, 1], log(0.2)), log(50))
            self.mean[i, 2] = min(max(self.mean[i, 2], log(0.1)), self.mean[i, 1])
            r_z_r_y_r_z(self.R, self.mean[i,3:])
            cblas_dscal(3, -1, &self.R[0, 0], 3)
            cblas_dscal(3, -1, &self.R[0, 2], 3)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 1, 1, &self.R[0, 0], 3, &self.R[0, 0], 3, 0,
                        &self.A[i, 0, 0], 3)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 1, -1, &self.R[1, 0], 3, &self.R[1, 0], 3, 1,
                        &self.A[i, 0, 0], 3)
            cblas_dscal(9, exp(self.mean[i,2]), &self.A[i, 0,0], 1)
            cblas_dcopy(3, &self.R[2, 0], 1, &self.mu[i, 0], 1)
            self.l_k_b[i, 0] = self.mean[i, 0]
            self.l_k_b[i, 1] = exp(self.mean[i, 1])
            self.l_k_b[i, 2] = exp(self.mean[i, 2])

        self.loss = self.mean[0, 4]
        self.prob.calculate_probabilities_sampled_bingham(self.mu, old_dir, self.A, self.l_k_b)
        cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0], 1)
        return info

cdef class UKFBinghamQuatAlt(Interpolation):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities prob, **kwargs):
        self.kappas = np.zeros((kwargs['dim_model'] // 6, ), dtype=DTYPE)
        self.weights = np.zeros((kwargs['dim_model'] // 6, ), dtype=DTYPE)
        dim_model = kwargs['dim_model']
        kwargs['dim_model'] = 6
        self.num_kalman = dim_model//6
        self.mean = np.zeros((dim_model//6,7), dtype=np.float64)
        self.mlinear  = np.zeros((8,kwargs['data'].shape[3]), dtype=np.float64) ##  Shpuld be always 8. For edges of cube.
        self.P = np.zeros((dim_model//6, 6, 6), dtype=np.float64)
        self.y = np.zeros((dim_model//6, kwargs['data'].shape[3]), dtype=DTYPE)
        self.res = np.zeros((kwargs['data'].shape[3]), dtype=np.float64)
        self._model1 = BinghamQuatModel(vector_field=vector_field[:, 0:1], **kwargs)
        self._kalman1 = KalmanQuat(kwargs['data'].shape[3], kwargs['dim_model'], self._model1)
        self._model2 = BinghamQuatModel(vector_field=vector_field[:, 1:2], **kwargs)
        self._kalman2 = KalmanQuat(kwargs['data'].shape[3], kwargs['dim_model'], self._model2)
        self.data = kwargs['data']
        self.angles  = np.zeros((3,), dtype=DTYPE)
        self.store_loss = kwargs['store_loss']
        self.A = np.zeros((dim_model//6, 3, 3))
        self.mu = np.zeros((dim_model//6, 3))
        self.l_k_b = np.zeros((dim_model//6, 3))
        self.R = np.zeros((3, 3), dtype=DTYPE)
        self.R2 = np.zeros((3,3))
        self._model = BinghamQuatModel(vector_field=vector_field, **kwargs)
        self.orth_both = np.zeros((3), dtype=DTYPE)
        self.orth_next= np.zeros((3), dtype=DTYPE)
        self.orth_old= np.zeros((3), dtype=DTYPE)
        self.newframe = np.zeros((4), dtype=DTYPE)
        self.oldframe= np.zeros((4), dtype=DTYPE)
        self.oldframe_inv= np.zeros((4), dtype=DTYPE)
        self.rot = np.zeros((4), dtype=DTYPE)
        self.conversion_matrix = esh_to_sym_matrix(kwargs['order'])
        self.conversion_matrix_inv = sym_to_esh_matrix(kwargs['order'])
        self.y_tensor = np.zeros((dim_model//6, kwargs['data'].shape[3] +1, 1, 1, 1), dtype=DTYPE)
        self.best_fit = np.zeros((4,2,1), dtype=DTYPE)
        self.fit_matrix=np.zeros((kwargs['data'].shape[3],2))


#   cpdef int rotate_state
    cdef fit_weights(self):
        cdef double[:,:] matrix_mult = np.zeros((2,2))
        cdef double[:,:] matrix_mult_inv = np.zeros((2,2))
        for i in range(2):
            kappa = max(min(exp(self.mean[i,1]), 89), 0.1)
            beta = max(min(exp(self.mean[i,2]), kappa), 0.1)
            quat2ZYZ(self.angles, self.mean[i, 3:8])
            c_sh_rotate_real_coef_fast(&self.fit_matrix[0,i], self.fit_matrix.shape[1], &self._model.lookup_table1[<int> kappa * 10, <int> beta * 10, 0],
                                           1, self._model.order, &self.angles[0])

        res = nnls(self.fit_matrix, self.y[0])
        self.mean[0,0] = res[0][0]
        self.mean[1,0] = res[0][1]

    cdef void init_kalman(self):
        """
        This script first calculates the low-rank approximation for a given fODF. Then it calculates the hessian for the first peak and set initial
        beta and kappa values based on this. For the second peak first the residual of the bingham distribution is subtracted
        to reduce the bias in small crossing areas. Then the same fitting procedure is conducted. 
        The code is not very clean because we have to switch between tensor and SH basis. 
        """
        cdef double[:] y_holder = np.zeros((28))
        cdef double[:]  res_sh = np.zeros((28))
        mapping = [0, 5,4,3,2,1, 14,13,12,11,10,9,8,7,6, 27,26,25,24,23,22,21,20,19,18,17,16,15]
        # Interpolate current point

        #convert to tensor basis
        for i in range(28):
            y_holder[i] = self.y[0, mapping[i]]

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, self.conversion_matrix.shape[0], 1,
                    self.conversion_matrix.shape[1],
                    1, &self.conversion_matrix[0, 0], self.conversion_matrix.shape[1], &y_holder[0], 1, 0,
                    &self.y_tensor[0, 1, 0, 0, 0], 1)

        # do low-rank approximation

        cblas_dscal(8, 0, &self.best_fit[0, 0, 0], 1)

        self.y_tensor[0, 0, 0, 0, 0] = 1
        self.y_tensor[1, 0, 0, 0, 0] = 1
        approx_all_spherical(self.best_fit, self.y_tensor[0], 0, 0, 2, 0, 0)
        ## FIT first rank-1. First substract second rank-1 tensor

        # create the residual and norm them:
        hota_6o3d_sym_eval(self.res, self.best_fit[0, 1, 0], self.best_fit[1:, 1, 0])
        cblas_daxpy(self.res.shape[0], -1, &self.res[0], 1, &self.y_tensor[0, 1, 0, 0, 0], 1)
        scale = hota_6o3d_sym_s_form(self.y_tensor[0, 1:, 0, 0, 0], self.best_fit[1:, 0, 0])
        #print(scale)
        cblas_dscal(28, 1 / scale, &self.y_tensor[0, 1, 0, 0, 0], 1)

        # residual in sh
        self._model1.kinit(self.mean[0], self.point_index[:3], self.best_fit[1:, 0, 0], self.P[0],
                           self.y_tensor[0, :, 0, 0, 0])
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, self.conversion_matrix.shape[0], 1,
                    self.conversion_matrix.shape[1],
                    1, &self.conversion_matrix_inv[0, 0], self.conversion_matrix.shape[1], &self.res[0], 1, 0,
                    &res_sh[0], 1)
        # substra
        for i in range(28):
            self.y[0, i] -= res_sh[mapping[i]]

        kappa = max(min(exp(self.mean[0, 1]), 89), 0.1)
        beta = max(min(exp(self.mean[0, 2]), kappa), 0.1)
        quat2ZYZ(self.angles, self.mean[0, 3:8])
        c_sh_rotate_real_coef_fast(&self.fit_matrix[0, 0], self.fit_matrix.shape[1],
                                   &self._model.lookup_table1[<int> kappa * 10, <int> beta * 10, 0],
                                   1, self._model.order, &self.angles[0])

        res = nnls(self.fit_matrix[:, 0:1], self.y[0])
        #print(res)
        for i in range(28):
            self.y[1, i] -= res[0][0] * self.fit_matrix[i, 0]
        for i in range(28):
            y_holder[i] = self.y[1, mapping[i]]


        trilinear_v(self.point_index[:3], self.y[1], self.mlinear, self.data)

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, self.conversion_matrix.shape[0], 1,
                    self.conversion_matrix.shape[1],
                    1, &self.conversion_matrix[0, 0], self.conversion_matrix.shape[1], &y_holder[0], 1, 0,
                    &self.y_tensor[1, 1, 0, 0, 0], 1)

        scale = hota_6o3d_sym_s_form(self.y_tensor[1, 1:, 0, 0, 0], self.best_fit[1:, 1, 0])
        cblas_dscal(28, 1 / scale, &self.y_tensor[1, 1, 0, 0, 0], 1)

        self._model2.kinit(self.mean[1], self.point_index[:3], self.best_fit[1:, 1, 0], self.P[1],
                           self.y_tensor[1, :, 0, 0, 0])

        self.fit_weights()
        cblas_dcopy(4, &self.mean[0, 3], 1, &self._kalman1.c_quat[0], 1)
        cblas_dcopy(4, &self.mean[1, 3], 1, &self._kalman2.c_quat[0], 1)
        cblas_dcopy(3, &self.mean[0, 0], 1, &self._kalman1.c_mean[0], 1)
        cblas_dscal(3, 0, &self._kalman1.c_mean[3], 1)
        cblas_dcopy(3, &self.mean[1, 0], 1, &self._kalman2.c_mean[0], 1)
        cblas_dscal(3, 0, &self._kalman2.c_mean[3], 1)
        print(np.array(self._kalman1.c_quat))

    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *:
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
        cdef int i,j, info = 0
        cdef double base = 0, scale

        trilinear_v(self.point_index[:3], self.y[0], self.mlinear, self.data)
        trilinear_v(self.point_index[:3], self.y[1], self.mlinear, self.data)

        # If we are at the seed. Initialize the Kalmanfilter
        if restart == 0:
            self.init_kalman()
        # Run Kalmannfilter
        for i in range(self.num_kalman):
            for j in range(self.num_kalman):
                if i == j:
                    continue
                kappa = min(max(exp(self.mean[j, 1]), 0.1), 89)
                beta = min(max(exp(self.mean[j, 2]), 0.1), kappa)


                quat2ZYZ(self._model.angles, self.mean[j,3:])
                c_sh_rotate_real_coef_fast(&self.res[0], 1, &self._model.lookup_table1[<int> kappa * 10, <int> beta * 10, 0],
                                           1, self._model.order, &self._model.angles[0])

                cblas_daxpy(self.res.shape[0], -max(self.mean[j, 0], _lambda_min), &self.res[0], 1, &self.y[i,0], 1)
            if i == 0:
                info = self._kalman1.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])
            else:
                info = self._kalman2.update_kalman_parameters(self.mean[i], self.P[i], self.y[i])

        print("mean", np.array(self.mean))

        for i in range(self.num_kalman):
            self.mean[i, 0] = max(self.mean[i, 0], _lambda_min)
            self.mean[i, 1] = min(max(self.mean[i, 1], log(0.2)), log(89))
            self.mean[i, 2] = min(max(self.mean[i, 2], log(0.1)), self.mean[i, 1])
            quat2rot(self.R, self.mean[i, 3:])

            cblas_dscal(3, -1, &self.R[0, 0], 3)
            cblas_dscal(3, -1, &self.R[0, 2], 3)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 1, 1, &self.R[0, 0], 3, &self.R[0, 0], 3, 0,
                        &self.A[i, 0, 0], 3)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 1, -1, &self.R[1, 0], 3, &self.R[1, 0], 3, 1,
                        &self.A[i, 0, 0], 3)
            cblas_dscal(9, exp(self.mean[i,2]), &self.A[i, 0,0], 1)
            cblas_dcopy(3, &self.R[2, 0], 1, &self.mu[i, 0], 1)
            self.l_k_b[i, 0] = self.mean[i, 0]
            self.l_k_b[i, 1] = exp(self.mean[i, 1])
            self.l_k_b[i, 2] = exp(self.mean[i, 2])

            print(np.array(self.R))

        print('mu', np.array(self.mu))
        print('old_dir', np.array(old_dir))
        print('A', np.array(self.A))
        print('l_k_b', np.array(self.l_k_b))
        self.prob.calculate_probabilities_sampled_bingham(self.mu, old_dir, self.A, self.l_k_b)
        print('best fit', np.array(self.prob.best_fit))
        cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0], 1)



        return info


cdef class UKFMultiTensor(UKF):
    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
        super(UKFMultiTensor, self).__init__(vector_field, grid, probClass, **kwargs)


        #cblas_dcopy(self.mean.shape[0], &self.mean[0], 1, &self.tmpmean[0], 1)
    cdef int select_next_dir(self, int info, double[:] old_dir):
        for i in range(self._model.num_tensors):
            if cblas_dnrm2(3, &self.mean[5*i], 1) != 0:
                cblas_dscal(3, 1 / cblas_dnrm2(3, &self.mean[5*i], 1), &self.mean[5*i], 1)
            if cblas_ddot(3, &self.mean[5*i], 1, &old_dir[0],1) < 0:
                cblas_dscal(3, -1, &self.mean[5*i], 1)
            self.mean[5*i+3] = max(self.mean[5*i+3],_lambda_min)
            self.mean[5*i+4] = max(self.mean[5*i+4],_lambda_min)

        # Use alw
        for i in range(self._model.num_tensors):
            if fa(self.mean[5*i + 3],self.mean[5*i + 4],self.mean[5*i + 4]) > 0.15:
                self.best_dir[i] = self.mean[5*i: 5*i + 3]
            else:
                self.mean[5 * i + 4] = min(self.mean[5*i + 3], self.mean[5*i + 4])
                set_zero_vector(self.best_dir[i])
        self.prob.calculate_probabilities(self.best_dir, old_dir)
        self.next_dir = self.prob.best_fit
        return info


cdef class DeepReg(Interpolation):

    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
        super(DeepReg, self).__init__(vector_field, grid, probClass, **kwargs)
        self.optimizer = RegLowRank(kwargs['data'], kwargs['reference'], kwargs['mu'], kwargs['meta'])
        self.reference = kwargs['reference']
        print('ref', self.reference.shape)
        self.ref_dir = np.zeros((3,), dtype=DTYPE)
        self.low_rank = np.zeros((12, ), dtype=DTYPE)
        self.y = np.zeros((16,), dtype=DTYPE)
        self.ylinear  = np.zeros((8,16), dtype=np.float64)
        self.rlinear = np.zeros((8, 3), dtype=np.float64)
        self.mu = kwargs['mu']
        self.data = kwargs['data']
        self.reg = np.zeros(3)
        self.low = np.zeros(3)
        self.opt = np.zeros(3)
        self.angle = 0
        self.selected_lambda = 0
        print("mu", self.mu)
        print(self.data.shape)
    


    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *:
        if restart==0:
            cblas_dscal(12, 0, &self.low_rank[0], 1)
        self.point_world[:3] = point
        self.point_world[3] = 1
        #print(np.asarray(old_dir))
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)

     #   print(np.array(self.point_index))
        cdef int z, i, info = 0
        cdef int idx = 0
        # Interpolate current point

        trilinear_v(self.point_index[:3], self.y, self.ylinear, self.data)

 #       cblas_dcopy(16, &self.data[<int> self.point_index[0], <int> self.point_index[1], <int> self.point_index[2],0], 1, &self.y[0], 1)
        trilinear_v_amb(self.point_index[:3], self.ref_dir, self.rlinear, self.reference)
        #print(np.array(self.ref_dir))
        #self.ref_dir = self.reference[<int> self.point_index[0], <int> self.point_index[1], <int> self.point_index[2],:]
        #print(np.array(self.ref_dir), np.array(self.reference[<int> self.point_index[0] ,  <int> self.point_index[1], <int> self.point_index[2],:]))
        cdef double scale = cblas_dnrm2(3, &self.ref_dir[0], 1)
        if scale != 0:
            cblas_dscal(3, 1/scale, &self.ref_dir[0], 1)
        self.optimizer.optimize_tensor(np.array(self.y[1:]), self.low_rank, 0, np.array(self.ref_dir), 0)

        cdef double min_ang = 0
        if cblas_dnrm2(3, &self.ref_dir[0], 1) > 0:
            for i in range(3):
                if self.low_rank[4*i] < 0.1:
                    continue
                cblas_dscal(3, 0, &self.best_dir[i,0], 1)
                ang = fabs(cblas_ddot(3, &self.ref_dir[0], 1, &self.low_rank[4*i + 1], 1))
                #print(ang)
                if ang > min_ang:
                    min_ang = ang
                    idx = i
            #print(idx)
            cblas_dcopy(3, &self.low_rank[4*idx+1], 1, &self.low[0], 1)
        cdef np.ndarray[np.float64_t, ndim=2] tens = np.zeros((3,15), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] length = np.zeros(3, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] best_dir_approx = np.zeros((3,3), dtype=np.float64)
        if self.y[0] == 0:
            return -1

        # If we have a reference direction, select the closest to it. Otherwise select closest to last direction:
        #cblas_dcopy(16, &self.data[<int> self.point_index[0], <int> self.point_index[1], <int> self.point_index[2],0], 1, &self.y[0], 1)
        trilinear_v(self.point_index[:3], self.y, self.ylinear, self.data)
        #print(np.asarray(self.y)) 
        if cblas_dnrm2(3, &self.ref_dir[0], 1) > 0 and self.mu > 0:
            idx = self.optimizer.min_mapping_voxel(np.asarray(self.low_rank), np.asarray(self.ref_dir))
            idx = self.optimizer.optimize_tensor(np.asarray(self.y[1:]), self.low_rank, idx, np.asarray(self.ref_dir), self.mu)
     #  # else:
     #  #     idx = self.optimizer.min_mapping_voxel(np.asarray(self.low_rank), np.asarray(old_dir))
        min_ang = 0
        if cblas_dnrm2(3, &self.ref_dir[0], 1) > 0:
            for i in range(3):
                if self.low_rank[4*i] < 0.1:
                    continue
                cblas_dscal(3, 0, &self.best_dir[i,0], 1)
                ang = fabs(cblas_ddot(3, &self.ref_dir[0], 1, &self.low_rank[4*i + 1], 1))
                #print(ang)
                if ang > min_ang:
                    min_ang = ang
                    idx = i
            #print(idx)
            cblas_dcopy(3, &self.low_rank[4*idx+1], 1, &self.opt[0], 1)
        cblas_dcopy(3, &self.ref_dir[0], 1, &self.reg[0], 1)
        #print(np.asarray(old_dir))
        min_ang = 0
        if cblas_dnrm2(3, &self.ref_dir[0], 1) > 0:
            for i in range(3):
                if self.low_rank[4*i] < 0.1:
                    continue
                cblas_dscal(3, 0, &self.best_dir[i,0], 1)
                ang = fabs(cblas_ddot(3, &self.ref_dir[0], 1, &self.low_rank[4*i + 1], 1))
                #print(ang)
                if ang > min_ang:
                    min_ang = ang
                    idx = i
            ang = fabs(cblas_ddot(3, &old_dir[0], 1, &self.low_rank[4*idx + 1], 1))
            #print(idx)
            if min_ang < 0.5 or ang < 0.5:
                cblas_dscal(3, 0, &self.next_dir[0], 1)
                return info 
            else:
                self.selected_lambda= self.low_rank[4*idx]
                if min_ang > 1:
                    min_ang = 1
                self.angle = np.arccos(min_ang)/np.pi*180
                
                cblas_dcopy(3, &self.low_rank[4*idx+1], 1, &self.next_dir[0], 1)
            if cblas_ddot(3, &self.next_dir[0], 1, &old_dir[0], 1) < 0:
                cblas_dscal(3, -1, &self.next_dir[0], 1)
            #length = np.random.normal(0, 0.1, 3)
           # cblas_daxpy(3, 1, &length[0], 1, &self.next_dir[0], 1)
           # cblas_dscal(3, 1/cblas_dnrm2(3, &self.next_dir[0], 1), &self.next_dir[0], 1)
               
        else:
          for i in range(3):
              if self.low_rank[4*i] < 0.1:
                  cblas_dscal(4, 0, &self.low_rank[4*i], 1)


          self.prob.select_next_dir(self.low_rank, old_dir)
          cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0],1)
        return info


cdef class DeepLearned(Interpolation):
    def __cinit__(self, double[:,:,:,:,:] vector_field, int[:] grid, Probabilities probClass, **kwargs):
        super(DeepLearned, self).__init__(vector_field, grid, probClass, **kwargs)
        self.lrs = tuple(( kwargs['lr_model'],  )) #, kwargs['lr_model_reg'] ))
        self.lrs[0].eval()
        self.low_rank = np.zeros((3,3 ), dtype=DTYPE)
        self.y = np.zeros((16,), dtype=DTYPE)
        self.ylinear  = np.zeros((8,16), dtype=np.float64)
        self.rlinear = np.zeros((8, 3), dtype=np.float64)
        self.mu = kwargs['mu']
        self.data = kwargs['data']
        self.reg = np.zeros(3)
        self.low = np.zeros(3)
        self.opt = np.zeros(3)
        self.angle = 0
        self.selected_lambda = 0
 #       self.lrs[1].eval()



    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *:
        if restart==0:
            cblas_dscal(9, 0, &self.low_rank[0,0], 1)
        self.point_world[:3] = point
        self.point_world[3] = 1
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)

        cdef int z, i, info = 0
        cdef int idx = 0
        # Interpolate current point

        trilinear_v(self.point_index[:3], self.y, self.ylinear, self.data)

 #       cblas_dcopy(16, &self.data[<int> self.point_index[0], <int> self.point_index[1], <int> self.point_index[2],0], 1, &self.y[0], 1)
#        trilinear_v_amb(self.point_index[:3], self.ref_dir, self.rlinear, self.reference)
 #       cdef double scale = cblas_dnrm2(3, &self.ref_dir[0], 1)
  #      if scale != 0:
   #         cblas_dscal(3, 1/scale, &self.ref_dir[0], 1)
   #         output = self.lrs[1](torch.cat((torch.tensor(self.y[1:]), torch.tensor(self.ref_dir))).float()[None])
   #         min_ang = 0
   #         for i in range(3):
   #             ang  = fabs(np.dot(self.ref_dir, output[0, 3*i:3*(i+1)].cpu().detach().numpy()/np.linalg.norm(output[0, 3*i:3*(i+1)].cpu().detach().numpy())))
   #             if ang > min_ang:
   #                 min_ang = ang
   #                 idx = i
#
 #           self.next_dir = output[0, 3*idx:3*(idx+1)].detach().numpy().astype(np.float64)
  #          if cblas_ddot(3, &self.next_dir[0], 1, &old_dir[0], 1) < 0:
   #             cblas_dscal(3, -1, &self.next_dir[0], 1)
     #       print(np.array(self.next_dir))
    #    else:
        output = self.lrs[0](torch.tensor(self.y[1:]).float()[None])
        output = output[0].cpu().detach().numpy()
        for i in range(3):
            for j in range(3):
                self.low_rank[i, j] = output[3*i + j].astype(np.float64)


        self.prob.calculate_probabilities(self.low_rank, old_dir)
        cblas_dcopy(3, &self.prob.best_fit[0], 1, &self.next_dir[0],1)
        return info






cdef class TomReg(Interpolation):

    def __cinit__(self, double[:,:,:,:,:]  vector_field, int[:] grid, Probabilities probClass, **kwargs):
        super(TomReg, self).__init__(vector_field, grid, probClass, **kwargs)
        self.reference = kwargs['reference']
        self.ref_dir = np.zeros((3,), dtype=DTYPE)
        self.rlinear = np.zeros((8, 3), dtype=np.float64)

    cpdef int interpolate(self, double[:] point, double[:] old_dir, int restart) except *:
        #print("HELLO")
        self.point_world[:3] = point
        self.point_world[3] = 1
        #print(np.asarray(old_dir))
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4,4,1,&self.inv_trafo[0,0], 4, &self.point_world[0], 1, 0, &self.point_index[0],1)
#        Nprint(np.array(self.point_index))
        cdef int z, i, info = 0
        # Interpolate current point

        trilinear_v_amb(self.point_index[:3], self.ref_dir, self.rlinear, self.reference)
        cdef double scale = cblas_dnrm2(3, &self.ref_dir[0], 1)
        if scale != 0:
            cblas_dscal(3, 1/scale, &self.ref_dir[0], 1)
        else:
            return 1
        if cblas_ddot(3, &self.ref_dir[0], 1, &old_dir[0], 1) < 0:
            cblas_dscal(3, -1, &self.ref_dir[0], 1)
        cblas_dcopy(3, &self.ref_dir[0], 1, &self.next_dir[0],1)
        return info








