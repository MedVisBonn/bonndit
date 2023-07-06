#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True, warn.unused_results=True
from libc.math cimport sqrt, cos, sin
import numpy as np
from bonndit.utilc.blas_lapack cimport *
from bonndit.utilc.cython_helpers cimport cart2sphere


cdef int[:] order_4_mult = np.array([1.0, 4.0, 4.0, 6.0, 12.0, 6.0, 4.0, 12.0, 12.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.int32)

cdef int[:] order_6_mult = np.array([1.0, 6.0, 6.0, 15.0, 30.0, 15.0, 20.0, 60.0, 60.0, 20.0, 15.0, 60.0, 90.0, 60.0, 15.0, 6.0, 30.0, 60.0, 60.0, 30.0, 6.0, 1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], dtype=np.int32)

cdef double hota_4o3d_sym_tsp(double[:] a, double[:] b) nogil:
    return a[0]*b[0]+a[10]*b[10]+a[14]*b[14]+4*(a[1]*b[1]+a[2]*b[2]+a[6]*b[6]+a[9]*b[9]+a[11]*b[11]+a[13]*b[13])+6*(a[3]*b[3]+a[5]*b[5]+a[12]*b[12])+12*(a[4]*b[4]+a[7]*b[7]+a[8]*b[8])

cdef double hota_6o3d_sym_tsp(double[:] a, double[:] b) nogil:
    return (a[0]*b[0]+a[21]*b[21]+a[27]*b[27]+ 6*(a[1]*b[1]+a[2]*b[2] + a[15]*b[15]+a[20]*b[20] +
            a[22]*b[22]+a[26]*b[26])+ 15*(a[3]*b[3]+ a[5]* b[5]+ a[10]*b[10]+ a[14]* b[14]+
            a[23]*b[23]+a[25]*b[25])+ 30*(a[4]*b[4]+a[16]*b[16]+a[19]*b[19])+
            20*(a[6]*b[6]+a[9]*b[9]+a[24]*b[24]) + 60*(a[7]*b[7]+a[8]*b[8]+a[11]*b[11]+
            a[13]*b[13]+a[17]*b[17]+a[18]*b[18])+ 90*a[12]*b[12])

cdef double hota_4o3d_sym_norm(double[:] a) nogil:
    return sqrt(hota_4o3d_sym_tsp(a,a))


cdef double hota_6o3d_sym_norm(double[:] a) nogil:
    return sqrt(hota_6o3d_sym_tsp(a,a))

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


cdef void hota_6o3d_sym_eval(double[:] res, double s, double[:] points) nogil:
    cdef double v00, v01, v02, v11, v12, v22
    v00 = points[0]* points[0]
    v01 = points[0]* points[1]
    v02 = points[0]* points[2]

    v11 = points[1]* points[1]
    v12 = points[1]* points[2]
    v22 = points[2]* points[2]
    res[0] = s * v00 * v00 * v00
    res[1] = s * v00 * v00 * v01
    res[2] = s * v00 * v00 * v02

    res[3] = s * v00 * v00 * v11
    res[4] = s * v00 * v00 * v12
    res[5] = s * v00 * v00 * v22

    res[6] = s * v00 * v01 * v11
    res[7] = s * v00 * v01 * v12
    res[8] = s * v00 * v01 * v22

    res[9] = s * v00 * v02 * v22
    res[10] = s * v00 * v11 * v11
    res[11] = s * v00 * v11 * v12

    res[12] = s * v00 * v11 * v22
    res[13] = s * v00 * v12 * v22
    res[14] = s * v00 * v22 * v22

    res[15] = s * v01 * v11 * v11
    res[16] = s * v01 * v11 * v12
    res[17] = s * v01 * v11 * v22

    res[18] = s * v01 * v12 * v22
    res[19] = s * v01 * v22 * v22
    res[20] = s * v02 * v22 * v22

    res[21] = s * v11 * v11 * v11
    res[22] = s * v11 * v11 * v12
    res[23] = s * v11 * v11 * v22

    res[24] = s * v11 * v12 * v22
    res[25] = s * v11 * v22 * v22
    res[26] = s * v12 * v22 * v22

    res[27] = s * v22 * v22 * v22


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

cdef void hota_6o3d_sym_make_iso(double[:] res, double s) nogil:
    cdef int i
    for i in range(28):
        res[i] = 0
    res[0]=res[21]=res[27]=s
    res[3]=res[5]=res[10]=res[14]=res[23]=res[25]=0.2*s
    res[12]=s/15.0


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
    

cdef void  hota_6o3d_sym_v_form(double[:] res, double[:] A, double[:] v) nogil:
    cdef double v00, v01, v02, v11 ,v12, v22, v00000, v00001, v00002, v00011, v00012, v00022, v00111, v00112, v00122, \
                v00222, v01111, v01112, v01122, v01222, v02222, v11111, v11112, v11122, v11222, v12222, v22222
    v00=v[0]*v[0]
    v01=v[0]*v[1]
    v02=v[0]*v[2]
    v11=v[1]*v[1]
    v12=v[1]*v[2]
    v22=v[2]*v[2]
    v00000=v00*v00*v[0]
    v00001=v00*v00*v[1]
    v00002=v00*v00*v[2]
    v00011=v00*v01*v[1]
    v00012=v00*v01*v[2]
    v00022=v00*v02*v[2]
    v00111=v00*v11*v[1]
    v00112=v00*v11*v[2]
    v00122=v00*v12*v[2]
    v00222=v00*v22*v[2]
    v01111=v01*v11*v[1]
    v01112=v01*v11*v[2]
    v01122=v01*v12*v[2]
    v01222=v01*v22*v[2]
    v02222=v02*v22*v[2]
    v11111=v11*v11*v[1]
    v11112=v11*v11*v[2]
    v11122=v11*v12*v[2]
    v11222=v11*v22*v[2]
    v12222=v12*v22*v[2]
    v22222=v22*v22*v[2]
    res[0] = A[0]*v00000+ 5*A[1]*v00001+ 5*A[2]*v00002+ 10*A[3]*v00011+ 20*A[4]*v00012+ 10*A[5]*v00022+ 10*A[6 ]*v00111+ 30*A[7 ]*v00112+ 30*A[8 ]*v00122+ 10*A[9 ]*v00222+ 5*A[10]*v01111+ 20*A[11]*v01112+ 30*A[12]*v01122+ 20*A[13]*v01222+ 5*A[14]*v02222+ A[15]*v11111+ 5*A[16]*v11112+ 10*A[17]*v11122+ 10*A[18]*v11222+ 5*A[19]*v12222+ A[20]*v22222
    res[1] = A[1]*v00000+ 5*A[3]*v00001+ 5*A[4]*v00002+ 10*A[6]*v00011+ 20*A[7]*v00012+ 10*A[8]*v00022+ 10*A[10]*v00111+ 30*A[11]*v00112+ 30*A[12]*v00122+ 10*A[13]*v00222+ 5*A[15]*v01111+ 20*A[16]*v01112+ 30*A[17]*v01122+ 20*A[18]*v01222+ 5*A[19]*v02222+ A[21]*v11111+ 5*A[22]*v11112+ 10*A[23]*v11122+ 10*A[24]*v11222+ 5*A[25]*v12222+ A[26]*v22222
    res[2] = A[2]*v00000+ 5*A[4]*v00001+ 5*A[5]*v00002+ 10*A[7]*v00011+ 20*A[8]*v00012+ 10*A[9]*v00022+ 10*A[11]*v00111+ 30*A[12]*v00112+ 30*A[13]*v00122+ 10*A[14]*v00222+ 5*A[16]*v01111+ 20*A[17]*v01112+ 30*A[18]*v01122+ 20*A[19]*v01222+ 5*A[20]*v02222+ A[22]*v11111+ 5*A[23]*v11112+ 10*A[24]*v11122+ 10*A[25]*v11222+ 5*A[26]*v12222+ A[27]*v22222

cdef double hota_4o3d_mean(double[:] a) nogil:
    return 0.2*(a[0] + a[10] + a[14]) + 2*(a[3] + a[5] + a[12])

cdef double hota_6o3d_mean(double[:] a) nogil:
    return (a[0] + a[21] + a[27] + 3*(a[3] + a[5] + a[10] + a[14] + a[23] + a[25]) + 6*a[12])/7

cdef double hota_4o3d_sym_s_form(double[:] a, double[:] v) nogil:
    cdef double v00, v01, v02, v11, v12, v22
    v00=v[0]*v[0]
    v01=v[0]*v[1]
    v02=v[0]*v[2]
    v11=v[1]*v[1]
    v12=v[1]*v[2]
    v22=v[2]*v[2]
    return a[0]*v00*v00+4*a[1]*v00*v01+4*a[2]*v00*v02+6*a[3]*v00*v11+12*a[4]*v00*v12+6*a[5]*v00*v22+4*a[6]*v01*v11+12*a[7]*v01*v12+12*a[8]*v01*v22+4*a[9]*v02*v22+a[10]*v11*v11+4*a[11]*v11*v12+6*a[12]*v11*v22+4*a[13]*v12*v22+a[14]*v22*v22





cpdef void hota_6o3d_hessian_sh(double[:,:] ret, double[:,:] proj, double[:] fodf, double[:] point):
    ## Using T. Schultz formula!
    cdef double[:,:] ret3D = np.zeros((3,3)), ret3D1 = np.zeros((2,3))
    cdef double[:] sh_coord = np.zeros((2,))
    #damit x,y berechnen
    cart2sphere(sh_coord, point)
    proj[0,0] = cos(sh_coord[0])*cos(sh_coord[1])
    proj[1,0] = cos(sh_coord[0])*sin(sh_coord[1])
    proj[2,0] = -sin(sh_coord[0])
    proj[0,1] = sin(sh_coord[0]) *sin(sh_coord[1])
    proj[1,1] = sin(sh_coord[0])*cos(sh_coord[1])
    proj[2,1] = 0
    hota_6o3d_hessian(ret3D, fodf, point)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 2,3,3,1, &proj[0,0],2, &ret3D[0,0], 3, 0, &ret3D1[0,0],3)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 3, 1, &ret3D1[0,0], 3, &proj[0,0], 2, 0, &ret[0,0], 2)
    cdef double v = hota_6o3d_sym_s_form(fodf, point)
    ret[0,0] -= v
    ret[1,1] -= v



cpdef void hota_6o3d_hessian(double[:,:] ret, double[:] fodf, double[:] point):
    ## computed by sympy
    cdef double x = point[0], y = point[1], z = point[2]
    ret[0,0] = 30 * fodf[0] * x ** 4 + 30 * fodf[10] * y ** 4 + 120 * fodf[11] * y ** 3 * z + 180 * fodf[
        12] * y ** 2 * z ** 2 + 120 * fodf[13] * y * z ** 3 + 30 * fodf[14] * z ** 4 + 120 * fodf[
          1] * x ** 3 * y + 120 * fodf[2] * x ** 3 * z + 180 * fodf[3] * x ** 2 * y ** 2 + 360 * fodf[
          4] * x ** 2 * y * z + 180 * fodf[5] * x ** 2 * z ** 2 \
      + 120 * fodf[6] * x * y ** 3 + 360 * fodf[7] * x * y ** 2 * z + 360 * fodf[8] * x * y * z ** 2 + 120 * fodf[
          9] * x * z ** 3
    ret[0,1] = 120 * fodf[10] * x * y ** 3 + 360 * fodf[11] * x * y ** 2 * z + 360 * fodf[12] * x * y * z ** 2 + 120 * fodf[
          13] * x * z ** 3 + 30 * fodf[15] * y ** 4 + 120 * fodf[16] * y ** 3 * z + 180 * fodf[17] * y ** 2 * z ** 2 \
      + 120 * fodf[18] * y * z ** 3 + 30 * fodf[19] * z ** 4 + 30 * fodf[1] * x ** 4 + 120 * fodf[
          3] * x ** 3 * y + 120 * fodf[4] * x ** 3 * z + 180 * fodf[6] * x ** 2 * y ** 2 + 360 * fodf[
          7] * x ** 2 * y * z + 180 * fodf[8] * x ** 2 * z ** 2
    ret[0,2] = 120 * fodf[11] * x * y ** 3 + 360 * fodf[12] * x * y ** 2 * z + 360 * fodf[13] * x * y * z ** 2 \
      + 120 * fodf[14] * x * z ** 3 + 30 * fodf[16] * y ** 4 + 120 * fodf[17] * y ** 3 * z + 180 * fodf[
          18] * y ** 2 * z ** 2 + 120 * fodf[19] * y * z ** 3 + 30 * fodf[20] * z ** 4 + 30 * fodf[2] * x ** 4 + 120 * \
      fodf[4] * x ** 3 * y + 120 * fodf[5] * x ** 3 * z + 180 * fodf[7] * x ** 2 * y ** 2 + 360 * fodf[
          8] * x ** 2 * y * z \
      + 180 * fodf[9] * x ** 2 * z ** 2
    ret[1,0] = ret[0,1]
    ret[1,1] = 180 * fodf[10] * x ** 2 * y ** 2 + 360 * fodf[11] * x ** 2 * y * z + 180 * fodf[12] * x ** 2 * z ** 2 + 120 * \
         fodf[15] * x * y ** 3 + 360 * fodf[16] * x * y ** 2 * z \
         + 360 * fodf[17] * x * y * z ** 2 + 120 * fodf[18] * x * z ** 3 + 30 * fodf[21] * y ** 4 + 120 * fodf[
             22] * y ** 3 * z + 180 * fodf[23] * y ** 2 * z ** 2 + 120 * fodf[24] * y * z ** 3 + 30 * fodf[
             25] * z ** 4 + 30 * fodf[3] * x ** 4 + 120 * fodf[6] * x ** 3 * y + 120 * fodf[7] * x ** 3 * z
    ret[1,2] = 180 * fodf[11] * x ** 2 * y ** 2 \
         + 360 * fodf[12] * x ** 2 * y * z + 180 * fodf[13] * x ** 2 * z ** 2 + 120 * fodf[16] * x * y ** 3 + 360 * \
         fodf[17] * x * y ** 2 * z + 360 * fodf[18] * x * y * z ** 2 + 120 * fodf[19] * x * z ** 3 + 30 * fodf[
             22] * y ** 4 + 120 * fodf[23] * y ** 3 * z + 180 * fodf[24] * y ** 2 * z ** 2 + 120 * fodf[
             25] * y * z ** 3 + 30 * fodf[26] * z ** 4 \
         + 30 * fodf[4] * x ** 4 + 120 * fodf[7] * x ** 3 * y + 120 * fodf[8] * x ** 3 * z
    ret[2,0] = ret[0,2]
    ret[2,1] = ret[1,2]
    ret[2,2] = 180 * fodf[12] * x ** 2 * y ** 2 + 360 * fodf[13] * x ** 2 * y * z + 180 * fodf[14] * x ** 2 * z ** 2 + 120 * \
         fodf[17] * x * y ** 3 + 360 * fodf[18] * x * y ** 2 * z + 360 * fodf[19] * x * y * z ** 2 + 120 * fodf[
             20] * x * z ** 3 + 30 * fodf[23] * y ** 4 + 120 * fodf[24] * y ** 3 * z + 180 * fodf[25] * y ** 2 * z ** 2 \
         + 120 * fodf[26] * y * z ** 3 + 30 * fodf[27] * z ** 4 + 30 * fodf[5] * x ** 4 + 120 * fodf[
             8] * x ** 3 * y + 120 * fodf[9] * x ** 3 * z





cdef double hota_6o3d_sym_s_form(double[:] s, double[:] points) nogil:
    cdef double v00, v01, v02, v11, v12, v22
    v00 = points[0]* points[0]
    v01 = points[0]* points[1]
    v02 = points[0]* points[2]

    v11 = points[1]* points[1]
    v12 = points[1]* points[2]
    v22 = points[2]* points[2]
    return 1 * s[0] * v00 * v00 * v00 + 6 * s[1] * v00 * v00 * v01 + 6 * s[2] * v00 * v00 * v02 + 15 * s[3] * v00 * v00 * v11 + 30 * s[4] * v00 * v00 * v12 + 15* s[5] * v00 * v00 * v22 + 20 * s[6] * v00 * v01 * v11 + \
            60 * s[7] * v00 * v01 * v12 + 60 * s[8] * v00 * v01 * v22 + 20 * s[9] * v00 * v02 * v22 + 15 * s[10] * v00 * v11 * v11 +  60 * s[11] * v00 * v11 * v12 + 90 * s[12] * v00 * v11 * v22 +  60 * s[13] * v00 * v12 * v22 + \
            15 * s[14] * v00 * v22 * v22 +  6 * s[15] * v01 * v11 * v11 +  30 * s[16] * v01 * v11 * v12 +  60 * s[17] * v01 * v11 * v22 +  60 * s[18] * v01 * v12 * v22 +  30* s[19] * v01 * v22 * v22 +  6 * s[20] * v02 * v22 * v22 + \
            1 * s[21] * v11 * v11 * v11 +  6 * s[22] * v11 * v11 * v12 +  15 * s[23] * v11 * v11 * v22 +  20 *  s[24] * v11 * v12 * v22 +  15* s[25] * v11 * v22 * v22 +  6 * s[26] * v12 * v22 * v22 + 1* s[27] * v22 * v22 * v22