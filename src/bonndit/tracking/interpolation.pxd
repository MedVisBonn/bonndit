#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True



from .ItoW cimport Trafo
from .alignedDirection cimport Probabilities, TractSegGetter
from .kalman.model cimport AbstractModel, fODFModel, WatsonModel, BinghamModel, BinghamQuatModel
from .kalman.kalman cimport Kalman, KalmanQuat
from bonndit.directions.regLowRank cimport RegLowRank
from bonndit.models.LowRank import LowRankModel

cdef class Interpolation:
    cdef double[:,:,:,:,:] vector_field
    cdef double[:,:,:] cuboid
    cdef double[:,:]  floor_point
    cdef double[:,:] inv_trafo
    cdef double[:] point_index
    cdef double[:] point_world
    cdef double[:,:] best_dir
    cdef int u
    cdef int[:,:,:,:] cache
    cdef double chosen_angle
    cdef Probabilities prob
    cdef double selected_lambda
    cdef double angle
    cdef double loss
    cdef int best_ind
    cdef double[:] next_dir, vector
    cdef bint check_point(self, double[:])
    cdef bint main_dir(self, double[:]) # nogil
    cdef void calc_cube(self, double[:]) # nogil
    cdef void nearest_neigh(self, double[:]) # nogil
    cdef void set_vector(self, int, int) # nogil
    cpdef int interpolate(self, double[:], double[:], int) except * # nogil except *

cdef class FACT(Interpolation):
    cpdef int interpolate(self, double[:], double[:], int) except * # nogil except *


cdef class Trilinear(Interpolation):
    cdef double[:,:] array, x_array, new_best_dir
    cdef int[:,:] not_check
    cdef double[:] point
    cdef double[:,:,:] dir
#   cdef int[:,:,:,:] cache
    cdef int[:] floor
    cdef int[:] permutation
    cdef void set_array(self, int, int, int) # nogil
    cpdef int interpolate(self, double[:], double[:], int) except * # nogil except *
    cdef void set_new_poss(self) # nogil except *
    cpdef get_cache(self)
    cpdef set_cache(self, int[:,:,:,:])
    cpdef get_next_dir(self)
    cpdef best_dirp(self)
    cpdef set_best_dirp(self, double[:,:])
    cdef int kmeans(self, double[:]) # nogil except *
    cdef void permute(self, double[:]) # nogil except *

cdef class TrilinearFODF(Interpolation):
    cdef double[:,:,:,:] data
    cdef double[:] fodf
    cdef double[:] fodf1
    cdef double[:] empty
    cdef double sigma_1
    cdef double sigma_2
    cdef int inc
    cdef double[:] point_diff
    cdef double[:,:] trafo
    cdef double[:] dist
    cdef double[:] length
    cdef bint auto
    cdef double[:,:] best_dir_approx
    cdef double r
    cdef int rank
    cdef double[:,:] vlinear
    cdef int[:,:] neighbors
    cdef void trilinear(self, double[:] point) # nogil except *
    cdef void neigh(self, double[:] point) # nogil except *
    cpdef int interpolate(self, double[:] point, double[:] old_dir, int r) except * # nogil except *

cdef class UKF(Interpolation):
    cdef double[:] mean
    cdef double[:,:] P
    cdef double[:,:,:,:] data
    cdef double[:,:] mlinear
    cdef double[:] y
    cdef Kalman _kalman
    cdef AbstractModel _model
    cpdef int interpolate(self, double[:], double[:], int) except * # nogil except *
    cdef int select_next_dir(self, int, double[:])

cdef class UKFFodfAlt(Interpolation):
    cdef int num_kalman
    cdef double[:,:] mean
    cdef double[:,:] mlinear
    cdef double[:,:,:] P
    cdef double[:,:] y
    cdef double[:] res
    cdef double[:,:,:,:] data
    cdef Kalman _kalman1
    cdef fODFModel  _model1
    cdef Kalman _kalman2
    cdef fODFModel _model2
    cpdef int interpolate(self, double[:] , double[:] , int ) except *

cdef class UKFWatsonAlt(Interpolation):
    cdef double[:] kappas
    cdef double[:] weights
    cdef int num_kalman
    cdef double[:,:] mean
    cdef double[:,:] mlinear
    cdef double[:,:,:] P
    cdef double[:,:] y
    cdef double[:] res

    cdef double[:] res_copy
    cdef WatsonModel _model1
    cdef WatsonModel _model
    cdef Kalman _kalman1
    cdef WatsonModel _model2
    cdef Kalman _kalman2
    cdef double[:,:,:,:] data
    cdef double[:] rot_pysh_v
    cdef double[:] pysh_v
    cdef double[:] angles
    cdef bint store_loss

cdef class UKFFodf(UKF):
    cdef int select_next_dir(self, int, double[:]) # nogil except *

cdef class UKFWatson(UKF):
    cdef double[:] kappas
    cdef double[:] weights

    cdef WatsonModel _model1
    cdef bint store_loss
    cdef int select_next_dir(self, int, double[:]) # nogil except *

cdef class UKFBinghamAlt(Interpolation):
    cdef double[:] kappas
    cdef double[:] weights
    cdef int num_kalman
    cdef double[:,:] mean
    cdef double[:,:] mlinear
    cdef double[:,:,:] P
    cdef double[:,:] y
    cdef double[:] res
    cdef BinghamModel _model1
    cdef BinghamModel _model
    cdef Kalman _kalman1
    cdef BinghamModel _model2
    cdef Kalman _kalman2
    cdef double[:,:,:,:] data
    cdef double[:] rot_pysh_v
    cdef double[:] pysh_v
    cdef double[:] angles
    cdef bint store_loss
    cdef double[:,:,:] A
    cdef double[:,:] R
    cdef double[:,:] mu
    cdef double[:,:] l_k_b

cdef class UKFBinghamQuatAlt(Interpolation):
    cdef fit_weights(self)
    cdef double[:] kappas
    cdef double[:] weights
    cdef int num_kalman
    cdef double[:,:] mean
    cdef double[:,:] mlinear
    cdef double[:,:,:] P
    cdef double[:,:] y
    cdef double[:] res
    cdef BinghamQuatModel _model1
    cdef BinghamQuatModel _model
    cdef KalmanQuat _kalman1
    cdef BinghamQuatModel _model2
    cdef KalmanQuat _kalman2
    cdef double[:,:,:,:] data
    cdef double[:] rot_pysh_v
    cdef double[:] pysh_v
    cdef double[:] angles
    cdef bint store_loss
    cdef double[:,:,:] A
    cdef double[:,:] R
    cdef double[:,:] R2
    cdef double[:] test
    cdef double[:,:] mu
    cdef double[:,:] l_k_b
    cdef double[:] orth_both
    cdef double[:] orth_next
    cdef double[:] orth_old
    cdef double[:] newframe
    cdef double[:] oldframe
    cdef double[:] oldframe_inv
    cdef double[:] rot
    cdef double[:,:] conversion_matrix
    cdef double[:,:] conversion_matrix_inv
    cdef double[:,:,:,:, :] y_tensor
    cdef double[:,:,:] best_fit
    cdef double[:,:] fit_matrix
    cdef double[:,:] fit_matrix_rot
    cdef void init_kalman(self)




cdef class UKFBingham(UKF):
    cdef double[:,:,:] A
    cdef double[:,:] R
    cdef double[:,:] mu
    cdef double[:,:] l_k_b
    cdef BinghamModel _model1
    cdef bint store_loss
    cdef int select_next_dir(self, int, double[:]) # nogil except *


cdef class UKFMultiTensor(UKF):
    cdef int select_next_dir(self, int, double[:]) # nogil except *


cdef class DeepReg(Interpolation):
    cdef double[:,:,:,:] reference
    cdef RegLowRank optimizer
    cdef double[:] ref_dir
    cdef double[:] low_rank
    cdef double[:] y
    cdef double[:,:] ylinear
    cdef double[:,:] rlinear
    cdef double mu
    cdef double[:,:,:,:] data
    cdef double[:] reg
    cdef double[:] opt

    cdef double[:] low

cdef class DeepLearned(Interpolation):
    cdef RegLowRank optimizer
    cdef double[:] ref_dir
    cdef double[:,:] low_rank
    cdef double[:] y
    cdef double[:,:] ylinear
    cdef double[:,:] rlinear
    cdef double mu
    cdef double[:,:,:,:] data
    cdef double[:] reg
    cdef double[:] opt

    cdef double[:] low
    cdef tuple lrs 


cdef class TomReg(Interpolation):
    cdef double[:,:,:,:] reference
    cdef double[:] ref_dir
    cdef double[:,:] rlinear


#cdef class TomReg(Interpolation):
   # cdef double[:,:,:,:] reference
  #  cdef double[:,:] rlinear
  #  cdef double[:] ref_dir
#
