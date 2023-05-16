import numpy as np
from .blas_lapack cimport *
from libc.math cimport atan2, asin, cos, sin, acos, atan, fmax, fmin
DTYPE = np.float64
cdef double[:] empty_quat = np.zeros((4,), dtype=DTYPE), empty_quat2 = np.zeros((4,), dtype=DTYPE), empty_ret = np.zeros((4,), dtype=DTYPE)
cdef double[:] empty_quat1 = np.zeros((4, ), dtype=DTYPE)
cdef double[:] empty_quat3 = np.zeros((4, ), dtype=DTYPE)

cdef void MRP_H2R(double[:] ret, double[:]  quat, int ret_space, int quat_space) nogil except *:
    """
    Modified rotrigues parameters. Maps H to R3 
    """
    cblas_dscal(3, 0, &ret[0], ret_space)
    cblas_daxpy(3, 4/(1+quat[0]), &quat[1], quat_space, &ret[0], ret_space)

cdef void MRP_R2H(double[:] ret, double[:] point, int ret_space, int point_space) nogil except *:
   """
   Inverse mapping
   """
   cblas_dscal(4, 0, &ret[0], ret_space)
   #point[0] -= 1
   #point[1] -= 1
   #point[2] -= 1

   ret[0] = 16 - cblas_dnrm2(3, &point[0], point_space)**2
   cblas_daxpy(3, 8, &point[0], point_space, &ret[1], ret_space)
   cblas_dscal(4, 1/(16 + cblas_dnrm2(3, &point[0], point_space)**2), &ret[0], ret_space)

cdef void quatmul(double[:] ret, double[:] x, double[:] y) nogil except *:
    """
    Quaternions multiplication
    """
    ret[0] = x[0]*y[0] - x[1]*y[1] - x[2]*y[2] - x[3]*y[3]
    ret[1] = x[0]*y[1] + x[1]*y[0] + x[2]*y[3] - x[3]*y[2]
    ret[2] = x[0]*y[2] - x[1]*y[3] + x[2]*y[0] + x[3]*y[1]
    ret[3] = x[0]*y[3] + x[1]*y[2] - x[2]*y[1] + x[3]*y[0]

cdef void quat_inv(double[:] ret, double[:] q, int ret_space, int q_space) nogil except *:
    """
    quaternions inversion
    """

    cblas_dscal(4, 0, &ret[0], ret_space)
    ret[0] = q[0]
    cblas_daxpy(4, -1, &q[1], q_space, &ret[1], ret_space)

cdef void MPR_H2R_q(double[:] ret, double[:] quat, double[:] q, int ret_space, int q_space): # nogil except *:
    """
    rotate quaternion around q inverse and map then.
    """
    quat_inv(empty_quat, q, 1, q_space)
    quatmul(empty_quat2, empty_quat, quat)
    MRP_H2R(ret, empty_quat2, ret_space, 1)

cdef void MPR_R2H_q(double[:] ret, double[:] point, double[:] q, int point_space, int q_space ): # nogil except *:
    """
    rotate quaternion around q inverse and map then.
    """
    MRP_R2H(empty_quat2, point, 1, point_space)
    cblas_dcopy(4, &q[0], q_space, &empty_quat[0], 1)

    quatmul(ret, empty_quat, empty_quat2)

#cpdef void quat2XYZ(double[:] ret, double[:] quat) nogil except *:
#    cpdef double t0, t1, t2
#    t0 = 2* (quat[0] * quat[1] + quat[2] * quat[3])
#    t1 = 1 - 2* (quat[1] * quat[1] + quat[2] * quat[2])
#    ret[0] = atan2(t0,t1)
#    t2 = 2 * (quat[0] * quat[2] - quat[3]*quat[1])
#    t2 = fmax(fmin(t2, 1), -1)
#    ret[1] = asin(t2)
#    t0 = 2* (quat[0] * quat[3] + quat[2] * quat[1])
#    t1 = 1 - 2* (quat[3] * quat[3] + quat[2] * quat[2])
#    ret[2] = atan2(t0, t1)

cdef void XYZ2quat(double[:] ret, double[:] xyz) nogil except *:
    cblas_dscal(4, 0, &empty_quat1[0], 1)
    cblas_dscal(4, 0, &empty_quat2[0], 1)
    cblas_dscal(4, 0, &empty_quat3[0], 1)
    empty_quat1[0] = cos(xyz[0]/2)
    empty_quat1[1] = sin(xyz[0]/2)
    empty_quat2[0] = cos(xyz[1]/2)
    empty_quat2[2] = sin(xyz[1]/2)
    empty_quat3[0] = cos(xyz[2]/2)
    empty_quat3[3] = sin(xyz[2]/2)
    quatmul(empty_quat, empty_quat2, empty_quat1)
    quatmul(ret, empty_quat3, empty_quat)

#cpdef void XYZ2ZYZ(double[:] ret, double[:] xyz) nogil except *:
#    """ Changes from XYZ to ZYZ angels"""
#    # R[2,2] - TODO was eigentlich wenn der Wert 0 ist???
#    if  cos(xyz[0])*cos(xyz[1]) != 0:
#        # arccos(R[2,2])
#        ret[1] = acos(cos(xyz[0])*cos(xyz[1]))
#        # - arctan2(R[2,1], R[2,0]
#        ret[2] = - atan2(sin(xyz[0])*cos(xyz[2]) + sin(xyz[1])*sin(xyz[2])*cos(xyz[0]), sin(xyz[0])*sin(xyz[2]) - sin(xyz[1])*cos(xyz[0])*cos(xyz[2]))
#        # arctan2(R[1,2], R[0,2])
#        ret[0]  = atan2(-sin(xyz[0])*cos(xyz[1]), sin(xyz[1]))
#


#cpdef void ZYZ2XYZ(double[:] ret, double[:] zyz) nogil except *:
#    """Change from ZYZ angels to XYZ"""
#    # arctan( - R[1,2]/R[2,2])
#    ret[0] = atan(-sin(zyz[0])*sin(zyz[1])/ cos(zyz[1]))
#    # R[0,2]
#    ret[1] = asin(sin(zyz[1])*cos(zyz[0]))
#    # arctan( - R[0,1]/R[0,0])
#    ret[2] = atan((sin(zyz[0])*cos(zyz[2]) + sin(zyz[2])*cos(zyz[0])*cos(zyz[1]))/(-sin(zyz[0])*sin(zyz[2]) + cos(zyz[0])*cos(zyz[1])*cos(zyz[2])))

cdef void quat2ZYZ(double[:] ret, double[:] quat) nogil except *:
    """
    Take a quaternion and returns ZYZ euler angles:
    
    Parameters
    ----------
    
    ret : array_like
          return value first:
    quat : array_like
            quaternion second
    """
    cdef double r32 = 2 * (quat[2] * quat[3] - quat[0] * quat[1])
    cdef double r31 = 2 * (quat[1] * quat[3] + quat[0] * quat[2])
    cdef double r33 = quat[0] ** 2 - quat[1] ** 2 - quat[2] ** 2 + quat[3] ** 2
    cdef double r13 = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
    cdef double r23 = 2 * (quat[2] * quat[3] + quat[0] * quat[1])
    ret[2] = atan2(r32, r31)
    ret[0] = atan2(r23, -r13)
    ret[1] = atan2(r31*cos(ret[2]) + r32*sin(ret[2]), r33)


cdef void ZYZ2quat(double[:] ret, double[:] zyz) nogil except *:
    """
    Take ZYZ euler angles and convert them into a quaternion 
    TODO: Das direkt ausschreiben! # Works:
    """
    cblas_dscal(4, 0, &empty_quat1[0], 1)
    cblas_dscal(4, 0, &empty_quat2[0], 1)
    cblas_dscal(4, 0, &empty_quat3[0], 1)
    empty_quat1[0] = cos(zyz[0] / 2)
    empty_quat1[3] = sin(zyz[0] / 2)
    empty_quat2[0] = cos(zyz[1] / 2)
    empty_quat2[2] = sin(zyz[1] / 2)
    empty_quat3[0] = cos(zyz[2] / 2)
    empty_quat3[3] = sin(zyz[2] / 2)
    quatmul(empty_quat, empty_quat2, empty_quat1)
    quatmul(ret, empty_quat3, empty_quat)

cdef void quat2rot(double[:,:] R, double[:] quat) nogil except *:
    # Works:
    R[0, 0] = quat[0] ** 2 + quat[1] ** 2 - quat[2] ** 2 - quat[3] ** 2
    R[1, 0] =  2 * (quat[1] * quat[2] - quat[0] * quat[3])
    R[2, 0] =   2 * (quat[1] * quat[3] + quat[0] * quat[2])
    R[0, 1] =  2 * (quat[1] * quat[2] + quat[0] * quat[3])
    R[1, 1] = quat[0] ** 2 - quat[1] ** 2 + quat[2] ** 2 - quat[3] ** 2
    R[2, 1] =   2 * (quat[2] * quat[3] - quat[0] * quat[1])
    R[0, 2] =  2 * (quat[1] * quat[3] - quat[0] * quat[2])
    R[1, 2] =   2 * (quat[2] * quat[3] + quat[0] * quat[1])
    R[2, 2] =  quat[0] ** 2 - quat[1] ** 2 - quat[2] ** 2 + quat[3] ** 2

cdef void basis2quat(double[:] q, double[:] a, double[:] b, double[:] c):
    cdef double T = a[0] + b[1] + c[2];
    cdef double s
    if T > 0:
        s = sqrt(T + 1) * 2
        q[1] = (b[2] - c[1]) / s
        q[2] = (c[0] - a[2]) / s
        q[3] = (a[1] - b[0]) / s
        q[0] = 0.25 * s
    elif a[0] > b[1] and a[0] > c[2]:
        s = sqrt(1 + a[0] - b[1] - c[2]) * 2
        q[1] = 0.25 * s
        q[2] = (a[1] + b[0]) / s
        q[3] = (c[0] + a[2]) / s
        q[0] = (b[2] - c[1]) / s
    elif b[1] > c[2]:
        s = sqrt(1 + b[1] - a[0] - c[2]) * 2
        q[1] = (a[1] + b[0]) / s
        q[2] = 0.25 * s
        q[3] = (b[2] + c[1]) / s
        q[0] = (c[0] - a[2]) / s
    else:
        s = sqrt(1 + c[2] - a[0] - b[1]) * 2
        q[1] = (c[0] + a[2]) / s
        q[2] = (b[2] + c[1]) / s
        q[3] = 0.25 * s
        q[0] = (a[1] - b[0]) / s

    cblas_dscal(4, 1/cblas_dnrm2(4, &q[0], 1), &q[0], 1)

