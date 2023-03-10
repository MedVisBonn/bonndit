#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True
from libc.math cimport sqrt
import numpy as np


cdef int[:] order_4_mult = np.array([1.0, 4.0, 4.0, 6.0, 12.0, 6.0, 4.0, 12.0, 12.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.int32)

cdef double hota_4o3d_sym_tsp(double[:] a, double[:] b) nogil:
    return a[0]*b[0]+a[10]*b[10]+a[14]*b[14]+4*(a[1]*b[1]+a[2]*b[2]+a[6]*b[6]+a[9]*b[9]+a[11]*b[11]+a[13]*b[13])+6*(a[3]*b[3]+a[5]*b[5]+a[12]*b[12])+12*(a[4]*b[4]+a[7]*b[7]+a[8]*b[8])


cdef double hota_4o3d_sym_norm(double[:] a) nogil:
    return sqrt(hota_4o3d_sym_tsp(a,a))

cdef hota_8o3d_sym_eval_cons(double[:,:] tensors,double[:,:] points):
    multiplicity = np.diag(np.array([1, 8, 8, 28, 56, 28, 56, 168, 168, 56, 70, 280, 420, 280, 70, 56, 280, 560, 560,
                                     280, 56,
                                     28, 168, 420, 560, 420, 168, 28, 8, 56, 168, 280, 280, 168, 56, 8, 1, 8, 28, 56,
                                     70, 56,
                                     28, 8, 1]))
    v00 = np.multiply(points[0], points[0])
    v01 = np.multiply(points[0], points[1])
    v02 = np.multiply(points[0], points[2])
    s = 1
    res = np.zeros((45, points.shape[1]))
    v11 = np.multiply(points[1], points[1])
    v12 = np.multiply(points[1], points[2])
    v22 = np.multiply(points[2], points[2])
    res[0] = s * np.multiply(np.multiply(np.multiply(v00, v00), v00), v00)
    res[1] = s * np.multiply(np.multiply(np.multiply(v00, v00), v00), v01)
    res[2] = s * np.multiply(np.multiply(np.multiply(v00, v00), v00), v02)

    res[3] = s * np.multiply(np.multiply(np.multiply(v00, v00), v00), v11)
    res[4] = s * np.multiply(np.multiply(np.multiply(v00, v00), v00), v12)
    res[5] = s * np.multiply(np.multiply(np.multiply(v00, v00), v00), v22)

    res[6] = s * np.multiply(np.multiply(np.multiply(v00, v00), v01), v11)
    res[7] = s * np.multiply(np.multiply(np.multiply(v00, v00), v01), v12)
    res[8] = s * np.multiply(np.multiply(np.multiply(v00, v00), v01), v22)

    res[9] = s * np.multiply(np.multiply(np.multiply(v00, v00), v02), v22)
    res[10] = s *np.multiply(np.multiply(np.multiply(v00, v00), v11), v11)
    res[11] = s *np.multiply(np.multiply(np.multiply(v00, v00), v11), v12)

    res[12] = s *np.multiply(np.multiply(np.multiply(v00, v00), v11), v22)
    res[13] = s *np.multiply(np.multiply(np.multiply(v00, v00), v12), v22)
    res[14] = s *np.multiply(np.multiply(np.multiply(v00, v00), v22), v22)

    res[15] = s *np.multiply(np.multiply(np.multiply(v00, v01), v11), v11)
    res[16] = s *np.multiply(np.multiply(np.multiply(v00, v01), v11), v12)
    res[17] = s *np.multiply(np.multiply(np.multiply(v00, v01), v11), v22)

    res[18] = s *np.multiply(np.multiply(np.multiply(v00, v01), v12), v22)
    res[19] = s *np.multiply(np.multiply(np.multiply(v00, v01), v22), v22)
    res[20] = s *np.multiply(np.multiply(np.multiply(v00, v02), v22), v22)

    res[21] = s *np.multiply(np.multiply(np.multiply(v00, v11), v11), v11)
    res[22] = s *np.multiply(np.multiply(np.multiply(v00, v11), v11), v12)
    res[23] = s *np.multiply(np.multiply(np.multiply(v00, v11), v11), v22)

    res[24] = s *np.multiply(np.multiply(np.multiply(v00, v11), v12), v22)
    res[25] = s *np.multiply(np.multiply(np.multiply(v00, v11), v22), v22)
    res[26] = s *np.multiply(np.multiply(np.multiply(v00, v12), v22), v22)

    res[27] = s *np.multiply(np.multiply(np.multiply(v00, v22), v22), v22)
    res[28] = s *np.multiply(np.multiply(np.multiply(v01, v11), v11), v11)
    res[29] = s *np.multiply(np.multiply(np.multiply(v01, v11), v11), v12)

    res[30] = s *np.multiply(np.multiply(np.multiply(v01, v11), v11), v22)
    res[31] = s *np.multiply(np.multiply(np.multiply(v01, v11), v12), v22)
    res[32] = s *np.multiply(np.multiply(np.multiply(v01, v11), v22), v22)

    res[33] = s *np.multiply(np.multiply(np.multiply(v01, v12), v22), v22)
    res[34] = s *np.multiply(np.multiply(np.multiply(v01, v22), v22), v22)
    res[35] = s *np.multiply(np.multiply(np.multiply(v02, v22), v22), v22)

    res[36] = s *np.multiply(np.multiply(np.multiply(v11, v11), v11), v11)
    res[37] = s *np.multiply(np.multiply(np.multiply(v11, v11), v11), v12)
    res[38] = s *np.multiply(np.multiply(np.multiply(v11, v11), v11), v22)

    res[39] = s *np.multiply(np.multiply(np.multiply(v11, v11), v12), v22)
    res[40] = s *np.multiply(np.multiply(np.multiply(v11, v11), v22), v22)
    res[41] = s *np.multiply(np.multiply(np.multiply(v11, v12), v22), v22)

    res[42] = s *np.multiply(np.multiply(np.multiply(v11, v22), v22), v22)
    res[43] = s *np.multiply(np.multiply(np.multiply(v12, v22), v22), v22)
    res[44] = s *np.multiply(np.multiply(np.multiply(v22, v22), v22), v22)
    res = np.dot(multiplicity, res)
    return np.dot(tensors, res)


cdef void hota_8o3d_sym_eval(double[:] res, double s, double[:] points) nogil:
    cdef double v00, v01, v02, v11, v12, v22
    v00 = points[0]* points[0]
    v01 = points[0]* points[1]
    v02 = points[0]* points[2]

    v11 = points[1]* points[1]
    v12 = points[1]* points[2]
    v22 = points[2]* points[2]
    res[0] = s * v00 * v00 * v00 * v00
    res[1] = s * v00 * v00 * v00 * v01
    res[2] = s * v00 * v00 * v00 * v02

    res[3] = s * v00 * v00 * v00 * v11
    res[4] = s * v00 * v00 * v00 * v12
    res[5] = s * v00 * v00 * v00 * v22

    res[6] = s * v00 * v00 * v01 * v11
    res[7] = s * v00 * v00 * v01 * v12
    res[8] = s * v00 * v00 * v01 * v22

    res[9] = s * v00 * v00 * v02 * v22
    res[10] = s * v00 * v00 * v11 * v11
    res[11] = s * v00 * v00 * v11 * v12

    res[12] = s * v00 * v00 * v11 * v22
    res[13] = s * v00 * v00 * v12 * v22
    res[14] = s * v00 * v00 * v22 * v22

    res[15] = s * v00 * v01 * v11 * v11
    res[16] = s * v00 * v01 * v11 * v12
    res[17] = s * v00 * v01 * v11 * v22

    res[18] = s * v00 * v01 * v12 * v22
    res[19] = s * v00 * v01 * v22 * v22
    res[20] = s * v00 * v02 * v22 * v22

    res[21] = s * v00 * v11 * v11 * v11
    res[22] = s * v00 * v11 * v11 * v12
    res[23] = s * v00 * v11 * v11 * v22

    res[24] = s * v00 * v11 * v12 * v22
    res[25] = s * v00 * v11 * v22 * v22
    res[26] = s * v00 * v12 * v22 * v22

    res[27] = s * v00 * v22 * v22 * v22
    res[28] = s * v01 * v11 * v11 * v11
    res[29] = s * v01 * v11 * v11 * v12

    res[30] = s * v01 * v11 * v11 * v22
    res[31] = s * v01 * v11 * v12 * v22
    res[32] = s * v01 * v11 * v22 * v22

    res[33] = s * v01 * v12 * v22 * v22
    res[34] = s * v01 * v22 * v22 * v22
    res[35] = s * v02 * v22 * v22 * v22

    res[36] = s * v11 * v11 * v11 * v11
    res[37] = s * v11 * v11 * v11 * v12
    res[38] = s * v11 * v11 * v11 * v22

    res[39] = s * v11 * v11 * v12 * v22
    res[40] = s * v11 * v11 * v22 * v22
    res[41] = s * v11 * v12 * v22 * v22

    res[42] = s * v11 * v22 * v22 * v22
    res[43] = s * v12 * v22 * v22 * v22
    res[44] = s * v22 * v22 * v22 * v22


cdef void hota_4o3d_sym_eval(double[:] res, double s, double[:] v) nogil:
    cdef double v00,v01, v02, v11, v12, v22

    v00=v[0]*v[0]
    v01=v[0]*v[1]
    v02=v[0]*v[2]
    v11=v[1]*v[1]
    v12=v[1]*v[2]
    v22=v[2]*v[2]

    res[0]=s*v00*v00
    res[1]=s*v00*v01
    res[2]=s*v00*v02
    res[3]=s*v00*v11
    res[4]=s*v00*v12
    res[5]=s*v00*v22
    res[6]=s*v01*v11
    res[7]=s*v01*v12
    res[8]=s*v01*v22
    res[9]=s*v02*v22
    res[10]=s*v11*v11
    res[11]=s*v11*v12
    res[12]=s*v11*v22
    res[13]=s*v12*v22
    res[14]=s*v22*v22


cdef void hota_4o3d_sym_make_iso(double[:] res, double s) nogil:
    res[0]=res[10]=res[14]=s
    res[3]=res[5]=res[12]=s/3.0
    res[1]=res[2]=res[4]=res[6]=res[7]=res[8]=res[9]=res[11]=res[13]=0.0


cdef void hota_4o3d_sym_v_form(double[:] res, double[:] a, double[:]  v) nogil:
    cdef double v000, v001, v002, v011, v012, v022, v111, v112, v122, v222
    v000=v[0]*v[0]*v[0]
    v001=v[0]*v[0]*v[1]
    v002=v[0]*v[0]*v[2]
    v011=v[0]*v[1]*v[1]
    v012=v[0]*v[1]*v[2]
    v022=v[0]*v[2]*v[2]
    v111=v[1]*v[1]*v[1]
    v112=v[1]*v[1]*v[2]
    v122=v[1]*v[2]*v[2]
    v222=v[2]*v[2]*v[2]

    res[0] = a[0]*v000+a[6]*v111+a[9]*v222+6*a[4]*v012+3*(a[1]*v001+a[2]*v002+a[3]*v011+a[5]*v022+a[7]*v112+a[8]*v122)
    res[1] = a[1]*v000+a[10]*v111+a[13]*v222+6*a[7]*v012+3*(a[3]*v001+a[4]*v002+a[6]*v011+a[8]*v022+a[11]*v112+a[12]*v122)
    res[2] = a[2]*v000+a[11]*v111+a[14]*v222+6*a[8]*v012+3*(a[4]*v001+a[5]*v002+a[7]*v011+a[9]*v022+a[12]*v112+a[13]*v122)

cdef double hota_4o3d_mean(double[:] a) nogil:
    return 0.2*(a[0] + a[10] + a[14]) + 2*(a[3] + a[5] + a[12])

cdef double hota_4o3d_sym_s_form(double[:] a, double[:] v) nogil:
    cdef double v00, v01, v02, v11, v12, v22
    v00=v[0]*v[0]
    v01=v[0]*v[1]
    v02=v[0]*v[2]
    v11=v[1]*v[1]
    v12=v[1]*v[2]
    v22=v[2]*v[2]
    return a[0]*v00*v00+4*a[1]*v00*v01+4*a[2]*v00*v02+6*a[3]*v00*v11+12*a[4]*v00*v12+6*a[5]*v00*v22+4*a[6]*v01*v11+12*a[7]*v01*v12+12*a[8]*v01*v22+4*a[9]*v02*v22+a[10]*v11*v11+4*a[11]*v11*v12+6*a[12]*v11*v22+4*a[13]*v12*v22+a[14]*v22*v22
