cdef extern from "sh_rot.h":
    void SHRotateRealCoef(double*, double*, int, double*, double*)
    void map_dipy_to_pysh_o4(double*, double*)
    void map_pysh_to_dipy_o4(double*, double*)
    void sh_watson_coeffs(double, double*, int)