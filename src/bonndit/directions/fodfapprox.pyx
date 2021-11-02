#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True

# import numpy as np
import numpy as np
import psutil
from bonndit.utilc.lowrank cimport approx_initial
from bonndit.utilc.penalty_spherical cimport calc_norm_spherical, refine_average_spherical
from bonndit.utilc.cython_helpers cimport mult_with_scalar, sub_vectors, \
    set_zero_vector, set_zero_matrix, add_vectors
from tqdm import tqdm
cimport numpy as np
from cython.parallel cimport threadid

DTYPE = np.float64

#Algorithm 1: 2
cdef double set_initial_spherical(double[:,:] v, double[:,:] pen) nogil:
    cdef int i, l = pen.shape[1]
    for i in range(l):
        add_vectors(v[:,i], v[:,i], pen[:,i])

###
# Get sum of neighbourhood for each voxel. Divide by
###
cdef void get_neighbor_for_coor(double[:] nearest_fodf_sum, double[:,:,:,:]  fodf, int[:]
                                 coor, int[:,:] neighbors) nogil:

    cdef int index
    for index in range(neighbors.shape[0]):
        if fodf.shape[1] > coor[0] + neighbors[index,0] >= 0 and \
           fodf.shape[2] > coor[1] + neighbors[index,1] >= 0 and \
            fodf.shape[3] > coor[2] + neighbors[index,2] >= 0:
            add_vectors(nearest_fodf_sum[:],nearest_fodf_sum[:],fodf[:,coor[0] + neighbors[index,0],coor[1] \
                        + neighbors[index,1],coor[2] + neighbors[index,2]])
    mult_with_scalar(nearest_fodf_sum, 1/neighbors.shape[0], nearest_fodf_sum)




cpdef approx_all_spherical(double[:,:,:] output, double[:,:] data, double[:,:,:,:]  fodf, int nearest, double nu, int rank, verbose):
    """ This function calculates the best tensor approximation with spherical approximation as described in xy.

    Parameters
    ----------
    output The output
    data tensor data
    fodf TODO this can be replaced with a shape
    nearest integer. How many neighbours should be used for the calculation. Given a point x all neighbours with
                        norm(x,y) < nearest are used
    nu [0,inf) regularization strength
    run_all True or False. If True it regularizes. If False it calculates just the average

    Returns modifies the input data. The output is the resulting data.
    -------

    """
    ##Number of threads to allocate memory to each thread and prevent interference.
    cdef int thread_num = psutil.cpu_count()
    cdef int i, j, k, num = data.shape[1]
    ##Neighbors of each voxel with absolut norm leq nearest
    cdef int[:,:] neighbors = np.array([[i, j, k] for i in range(-nearest, nearest + 1) \
                                        for j in range(-nearest, nearest + 1) for k in
         range(-nearest, nearest + 1) if np.linalg.norm(np.array([i,j,k])) <= nearest], dtype=np.intc)
    cdef int neighbors_count = neighbors.shape[0]
   # print(neighbors_count)
    ##Set of coordinates
    cdef int[:,:] coordinates = np.array([[i, j, k] for i in range(fodf.shape[1]) for j in range(fodf.shape[2]) \
                                          for k in range(fodf.shape[3])], dtype=np.intc)

    cdef double[:,:] sum_data = np.zeros((16, thread_num), dtype=DTYPE)
   # cdef double[:,:,:] nearest_fodf = np.zeros([16, thread_num], dtype=DTYPE)
    #many helper variables
    cdef double[:, :, :] vs = np.zeros([4, 3, thread_num], dtype=DTYPE)
    cdef double[:,:] valsec = np.empty([1, thread_num], dtype=DTYPE), val= np.empty([1,thread_num], dtype=DTYPE), \
                     der = np.empty([3,thread_num],  dtype=DTYPE),
    cdef double[:,:] testv = np.empty([3,thread_num], dtype=DTYPE), \
                    anisoten = np.empty([15,thread_num],  dtype=DTYPE), \
                        isoten = np.empty([15, thread_num],dtype=DTYPE)
    cdef double[:] norm = np.zeros([thread_num], dtype=DTYPE)
    cdef double[:,:] res = np.empty([16,thread_num], dtype=DTYPE),
    cdef double[:, :, :] tens = np.zeros([3, 15, thread_num], dtype=DTYPE)
    cdef double[:, :, :] tens_average = np.zeros([3, 15, thread_num], dtype=DTYPE)
    cdef double[:, :, :] pen_act = np.empty([3, 3, thread_num], dtype=DTYPE), \
        three_matrix = np.empty([3, 3, thread_num], dtype=DTYPE)
    cdef double[:,:] three_vector = np.empty([3, thread_num], dtype=DTYPE) , one_vector = np.empty([1, thread_num], dtype=DTYPE)
    cdef double[:,:,:] three_vector_placeholder = np.empty([3,  5, thread_num], dtype=DTYPE)
    #print(1)
    for i in tqdm(range(num), disable=not verbose):
        ##get neighbourhood for each coordinate and save in the blocked thread memory.
       # set_zero_matrix(nearest_fodf[:,:, threadid()])
        set_zero_vector(sum_data[:, threadid()])
        get_neighbor_for_coor(sum_data[:, threadid()], fodf, coordinates[i],neighbors)
        if sum_data[0, threadid()] == 0.0 or fodf[0, coordinates[i,0], coordinates[i,1], coordinates[i,2]] == 0.0:
            continue

        ##Calc the Average
        approx_initial(output[0, :, i] , output[1:, :, i] , tens[:,:,  threadid()], sum_data[1:,  threadid()],
                            rank,  valsec[:, threadid()],
                               val[:, threadid()],
                               der[:, threadid()], testv[:, threadid()], anisoten[:, threadid()], isoten[:, threadid()])



#cpdef approx_all(fodf):
#    num = fodf.shape
#    for i in range(num):
#        if
#            _candidates_3d_d
