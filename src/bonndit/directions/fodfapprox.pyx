#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True

# import numpy as np
import numpy as np
import psutil
import scipy
from bonndit.utilc.hota cimport hota_4o3d_sym_eval, order_4_mult
from bonndit.utilc.lowrank cimport approx_initial
from bonndit.utilc.penalty_spherical cimport calc_norm_spherical, refine_average_spherical
from bonndit.utilc.cython_helpers cimport mult_with_scalar, sub_vectors, \
    set_zero_vector, set_zero_matrix, add_vectors
from tqdm import tqdm
#cimport numpy as np

DTYPE = np.float64

#Algorithm 1: 2
cdef double set_initial_spherical(double[:,:] v, double[:,:] pen) : # nogil:
    cdef int i, l = pen.shape[1]
    for i in range(l):
        add_vectors(v[:,i], v[:,i], pen[:,i])

###
# Get sum of neighbourhood for each voxel. Divide by
###
cdef void get_neighbor_for_coor(double[:] nearest_fodf_sum, double[:,:,:,:]  fodf, int[:]
                                 coor, int[:,:] neighbors) : # nogil:

    cdef int index
    for index in range(neighbors.shape[0]):
        if fodf.shape[1] > coor[0] + neighbors[index,0] >= 0 and \
           fodf.shape[2] > coor[1] + neighbors[index,1] >= 0 and \
            fodf.shape[3] > coor[2] + neighbors[index,2] >= 0:
            add_vectors(nearest_fodf_sum[:],nearest_fodf_sum[:],fodf[:,coor[0] + neighbors[index,0],coor[1] \
                        + neighbors[index,1],coor[2] + neighbors[index,2]])
    mult_with_scalar(nearest_fodf_sum, 1/neighbors.shape[0], nearest_fodf_sum)




cpdef approx_all_spherical(double[:,:,:] output, double[:,:,:,:]  fodf, int nearest, double nu, int rank, int init, verbose):
    """ This function calculates the best tensor approximation with spherical approximation as described in xy.

    Parameters
    ----------
    output The output
    data tensor data
    fodf TODO this can be replaced with a shape
    nearest integer. How many neighbours should be used for the calculation. Given a point x all neighbours with norm(x,y) < nearest are used
    nu [0,inf) regularization strength
    run_all True or False. If True it regularizes. If False it calculates just the average

    Returns modifies the input data. The output is the resulting data.
    -------

    """
    ##Number of threads to allocate memory to each thread and prevent interference.
    cdef int i, j, k, num =	fodf.shape[1] * fodf.shape[2] * fodf.shape[3]
    ##Neighbors of each voxel with absolut norm leq nearest
    cdef int[:,:] neighbors = np.array([[i, j, k] for i in range(-nearest, nearest + 1) \
                                        for j in range(-nearest, nearest + 1) for k in
         range(-nearest, nearest + 1) if np.linalg.norm(np.array([i,j,k])) <= nearest], dtype=np.intc)
    cdef int neighbors_count = neighbors.shape[0]
   # print(neighbors_count)
    ##Set of coordinates
    cdef int[:,:] coordinates = np.array([[i, j, k] for i in range(fodf.shape[1]) for j in range(fodf.shape[2]) \
                                          for k in range(fodf.shape[3])], dtype=np.intc)

    cdef double[:] sum_data = np.zeros((16), dtype=DTYPE)
   # cdef double[:,:,:] nearest_fodf = np.zeros([16, thread_num], dtype=DTYPE)
    #many helper variables
    cdef double[:,  :] vs = np.zeros([4, 3], dtype=DTYPE)
    cdef double[:] valsec = np.empty([1,], dtype=DTYPE), val= np.empty([1,], dtype=DTYPE), \
                     der = np.empty([3,],  dtype=DTYPE),
    cdef double[:] testv = np.empty([3,], dtype=DTYPE), \
                    anisoten = np.empty([15,],  dtype=DTYPE), \
                        isoten = np.empty([15, ],dtype=DTYPE)
    cdef double[:] norm = np.zeros([1,], dtype=DTYPE)
    cdef double[:] res = np.empty([16,], dtype=DTYPE),
    cdef double[:, :] tens = np.zeros([3, 15,], dtype=DTYPE)
    cdef double[:, :] tens_average = np.zeros([3, 15,], dtype=DTYPE)
    cdef double[:, :] pen_act = np.empty([3, 3], dtype=DTYPE), \
        three_matrix = np.empty([3, 3], dtype=DTYPE)
    cdef double[:] three_vector = np.empty([3,], dtype=DTYPE) , one_vector = np.empty([1,], dtype=DTYPE)
    cdef double[:,:] three_vector_placeholder = np.empty([3,  5,], dtype=DTYPE)
    #print(1)
    for i in tqdm(range(num), disable=not verbose):
        ##get neighbourhood for each coordinate and save in the blocked thread memory.
       # set_zero_matrix(nearest_fodf[:,:, threadid()])
        set_zero_vector(sum_data[:])
        get_neighbor_for_coor(sum_data[:], fodf, coordinates[i],neighbors)
        bsingle =  np.array(sum_data[:])
        if sum_data[0] == 0.0 or fodf[0, coordinates[i,0], coordinates[i,1], coordinates[i,2]] == 0.0:
            continue

        # Initialize the tens
        if init:
            for j in range(rank):
                hota_4o3d_sym_eval(tens[j, :], 1, output[1:, j, i])
            # copy to account for weighting
            A = []
            b = []
            for j in range(15):
                b = b + [bsingle[j] for k in range(order_4_mult[j])]
                A = A + [tens[:, j] for k in range(order_4_mult[j])]
            b = np.array(b)
            A = np.array(A)
            # least squares
            result = scipy.optimize.least_squares(minimize, np.array(output[0, :, i]), args=(A, b), bounds=([0,0,0], np.inf))
            x = result.x
           # print(result)
            mult_with_scalar(output[0, :, i],  1, x)
            if min(x) < 0:
                print(result)
            #print(x)
        # sub from the data. To initialize the iterative process!
        for j in range(rank):
            hota_4o3d_sym_eval(tens[j, :], output[0, j, i], output[1:, j, i])
            sub_vectors(sum_data[1:], sum_data[1:], tens[j,:])






        ##Calc the Average
        approx_initial(output[0, :, i] , output[1:, :, i] , tens[:,:], sum_data[1:],
                            rank,  valsec[:],
                               val[:],
                               der[:], testv[:], anisoten[:], isoten[:])




def minimize(x, A, b):
    return A@x - b
