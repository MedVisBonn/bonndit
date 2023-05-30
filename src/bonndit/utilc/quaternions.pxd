cdef void MRP_H2R(double[:], double[:], int, int) nogil except *
cdef void MRP_R2H(double[:], double[:], int, int) nogil except *
cdef void quatmul(double[:], double[:], double[:]) nogil except *
cdef void quat_inv(double[:], double[:], int, int) nogil except *
cdef void MPR_H2R_q(double[:],double[:],double[:], int, int) #nogil except *
cdef void MPR_R2H_q(double[:],double[:],double[:], int, int) #nogil except *
cdef void quat2ZYZ(double[:], double[:]) nogil except *
cdef void ZYZ2quat(double[:], double[:]) nogil except *
cdef void quat2rot(double[:,:], double[:]) nogil except *
cdef void basis2quat(double[:], double[:], double[:], double[:])
