import numpy as np
import os
from scipy.special import dawsn
import pyshtools as pysh

# fodf signal generation
def watson_fodf_signals_generator(x, n, dj, rank1conv, only_one_peak, peak_number, fixed_weight, weight_number, lmax):
    cimax = int((lmax*(lmax+1))/2+lmax+1)
    predicted_watson = np.zeros(cimax)

    for i in range(n):
        if only_one_peak:
            i = peak_number

        if fixed_weight:
            weight = weight_number
        else:
            weight = np.abs(x[i*4]) # weight of the watson distribution
        kappa = np.abs(x[i*4+1]) # kappa value for watson
        beta = x[i*4+2] # second euler angle in radians for watson peak
        gamma = x[i*4+3] # third euler angle in radians for watson peak

        dipy_coeffs = sh_watson_coeffs_o8(kappa)[:cimax]

        if rank1conv:
            if lmax == 4:
                dipy_coeffs[0] *= rank1_rh_o4[0]
                dipy_coeffs[3] *= rank1_rh_o4[1]
                dipy_coeffs[10] *= rank1_rh_o4[2]
            elif lmax == 6:
                dipy_coeffs[0] *= rank1_rh_o6[0]
                dipy_coeffs[3] *= rank1_rh_o6[1]
                dipy_coeffs[10] *= rank1_rh_o6[2]
                dipy_coeffs[21] *= rank1_rh_o6[3]
            elif lmax == 8:
                dipy_coeffs[0] *= rank1_rh_o8[0]
                dipy_coeffs[3] *= rank1_rh_o8[1]
                dipy_coeffs[10] *= rank1_rh_o8[2]
                dipy_coeffs[21] *= rank1_rh_o8[3]
                dipy_coeffs[36] *= rank1_rh_o8[4]

        # compute watson sh
        if lmax == 4:
            pysh_watson_coeffs = map_dipy_to_pysh_o4(dipy_coeffs)
        elif lmax == 6:
            pysh_watson_coeffs = map_dipy_to_pysh_o6(dipy_coeffs)
        elif lmax == 8:
            pysh_watson_coeffs = map_dipy_to_pysh_o8(dipy_coeffs)

        # rotate watson sh
        angles = np.array([0., beta, gamma])
        coeffs_rot = pysh.rotate.SHRotateRealCoef(pysh_watson_coeffs, angles, dj)

        # map to dipy notation
        if lmax == 4:
            predicted_watson += weight * map_pysh_to_dipy_o4(coeffs_rot)
        elif lmax == 6:
            predicted_watson += weight * map_pysh_to_dipy_o6(coeffs_rot)
        elif lmax == 8:
            predicted_watson += weight * map_pysh_to_dipy_o8(coeffs_rot)

        if only_one_peak:
            return predicted_watson

    return predicted_watson

def sh_watson_coeffs_o8(kappa):
    sqrt_k = np.sqrt(kappa)
    Fk = dawsn(sqrt_k)
    kappa_sq = kappa**2
    kappa_sq3 = kappa**3
    c_0 = 1 / (4*np.pi) * 2 * np.sqrt(np.pi)
    c_2 = 1 / (4*np.pi) * (3 / (sqrt_k*Fk) - 3/kappa -2)*np.pi * np.sqrt(5 / (4*np.pi))
    c_4 = 1 / (4*np.pi) * (5*sqrt_k*(2*kappa-21)/Fk + 12*kappa_sq + 60*kappa + 105)*1/(8*kappa_sq)*np.pi * np.sqrt(9 / (4*np.pi))
    c_6 = 1 / (4*np.pi) * (21*sqrt_k*(4*(kappa-5)*kappa+165) / Fk - 5*(8*kappa_sq3+84*kappa_sq+378*kappa+693))*1/(32*kappa_sq3)*np.pi * np.sqrt(13 / (4*np.pi))
    c_8 = 1 / (4*np.pi) * np.pi / (512.0 * kappa**4) * ((3*sqrt_k*(2*kappa*(2*kappa*(62*kappa-1925)+15015)-225225.0)) / Fk + 35*(8*kappa*(kappa*(2*kappa*(kappa + 18)+297)+1287)+19305)) * np.sqrt(19 / (4*np.pi))
    sh_wat = np.array([c_0, 0, 0, c_2, 0, 0, 0, 0, 0, 0, c_4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c_6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c_8, 0, 0, 0, 0, 0, 0, 0, 0])
    return sh_wat

def map_pysh_to_dipy_o8(sh):
    mapped_sh = np.array([sh[0,0,0],sh[1,2,2],sh[1,2,1],sh[0,2,0],sh[0,2,1],sh[0,2,2],sh[1,4,4],sh[1,4,3],sh[1,4,2],
                          sh[1,4,1],sh[0,4,0],sh[0,4,1],sh[0,4,2],sh[0,4,3],sh[0,4,4],sh[1,6,6],sh[1,6,5],sh[1,6,4],
                          sh[1,6,3],sh[1,6,2],sh[1,6,1],sh[0,6,0],sh[0,6,1],sh[0,6,2],sh[0,6,3],sh[0,6,4],sh[0,6,5],
                          sh[0,6,6],sh[1,8,8],sh[1,8,7],sh[1,8,6],sh[1,8,5],sh[1,8,4],sh[1,8,3],sh[1,8,2],sh[1,8,1],
                          sh[0,8,0],sh[0,8,1],sh[0,8,2],sh[0,8,3],sh[0,8,4],sh[0,8,5],sh[0,8,6],sh[0,8,7],sh[0,8,8]])
    return mapped_sh

def map_pysh_to_dipy_o6(sh):
    mapped_sh = np.array([sh[0,0,0],sh[1,2,2],sh[1,2,1],sh[0,2,0],sh[0,2,1],sh[0,2,2],sh[1,4,4],sh[1,4,3],sh[1,4,2],
                          sh[1,4,1],sh[0,4,0],sh[0,4,1],sh[0,4,2],sh[0,4,3],sh[0,4,4],sh[1,6,6],sh[1,6,5],sh[1,6,4],
                          sh[1,6,3],sh[1,6,2],sh[1,6,1],sh[0,6,0],sh[0,6,1],sh[0,6,2],sh[0,6,3],sh[0,6,4],sh[0,6,5],sh[0,6,6]])
    return mapped_sh

def map_pysh_to_dipy_o4(sh):
    mapped_sh = np.array([sh[0,0,0],sh[1,2,2],sh[1,2,1],sh[0,2,0],sh[0,2,1],sh[0,2,2],sh[1,4,4],sh[1,4,3],sh[1,4,2],
                          sh[1,4,1],sh[0,4,0],sh[0,4,1],sh[0,4,2],sh[0,4,3],sh[0,4,4]])
    return mapped_sh

def map_dipy_to_pysh_o8(sh):
    mapped_sh = np.zeros((2, 9, 9))
    mapped_sh[0,0,0] = sh[0]
    mapped_sh[1,2,2] = sh[1]
    mapped_sh[1,2,1] = sh[2]
    mapped_sh[0,2,0] = sh[3]
    mapped_sh[0,2,1] = sh[4]
    mapped_sh[0,2,2] = sh[5]
    mapped_sh[1,4,4] = sh[6]
    mapped_sh[1,4,3] = sh[7]
    mapped_sh[1,4,2] = sh[8]
    mapped_sh[1,4,1] = sh[9]
    mapped_sh[0,4,0] = sh[10]
    mapped_sh[0,4,1] = sh[11]
    mapped_sh[0,4,2] = sh[12]
    mapped_sh[0,4,3] = sh[13]
    mapped_sh[0,4,4] = sh[14]
    mapped_sh[1,6,6] = sh[15]
    mapped_sh[1,6,5] = sh[16]
    mapped_sh[1,6,4] = sh[17]
    mapped_sh[1,6,3] = sh[18]
    mapped_sh[1,6,2] = sh[19]
    mapped_sh[1,6,1] = sh[20]
    mapped_sh[0,6,0] = sh[21]
    mapped_sh[0,6,1] = sh[22]
    mapped_sh[0,6,2] = sh[23]
    mapped_sh[0,6,3] = sh[24]
    mapped_sh[0,6,4] = sh[25]
    mapped_sh[0,6,5] = sh[26]
    mapped_sh[0,6,6] = sh[27]
    mapped_sh[1,8,8] = sh[28]
    mapped_sh[1,8,7] = sh[29]
    mapped_sh[1,8,6] = sh[30]
    mapped_sh[1,8,5] = sh[31]
    mapped_sh[1,8,4] = sh[32]
    mapped_sh[1,8,3] = sh[33]
    mapped_sh[1,8,2] = sh[34]
    mapped_sh[1,8,1] = sh[35]
    mapped_sh[0,8,0] = sh[36]
    mapped_sh[0,8,1] = sh[37]
    mapped_sh[0,8,2] = sh[38]
    mapped_sh[0,8,3] = sh[39]
    mapped_sh[0,8,4] = sh[40]
    mapped_sh[0,8,5] = sh[41]
    mapped_sh[0,8,6] = sh[42]
    mapped_sh[0,8,7] = sh[43]
    mapped_sh[0,8,8] = sh[44]

    return mapped_sh

def map_dipy_to_pysh_o6(sh):
    mapped_sh = np.zeros((2, 7, 7))
    mapped_sh[0,0,0] = sh[0]
    mapped_sh[1,2,2] = sh[1]
    mapped_sh[1,2,1] = sh[2]
    mapped_sh[0,2,0] = sh[3]
    mapped_sh[0,2,1] = sh[4]
    mapped_sh[0,2,2] = sh[5]
    mapped_sh[1,4,4] = sh[6]
    mapped_sh[1,4,3] = sh[7]
    mapped_sh[1,4,2] = sh[8]
    mapped_sh[1,4,1] = sh[9]
    mapped_sh[0,4,0] = sh[10]
    mapped_sh[0,4,1] = sh[11]
    mapped_sh[0,4,2] = sh[12]
    mapped_sh[0,4,3] = sh[13]
    mapped_sh[0,4,4] = sh[14]
    mapped_sh[1,6,6] = sh[15]
    mapped_sh[1,6,5] = sh[16]
    mapped_sh[1,6,4] = sh[17]
    mapped_sh[1,6,3] = sh[18]
    mapped_sh[1,6,2] = sh[19]
    mapped_sh[1,6,1] = sh[20]
    mapped_sh[0,6,0] = sh[21]
    mapped_sh[0,6,1] = sh[22]
    mapped_sh[0,6,2] = sh[23]
    mapped_sh[0,6,3] = sh[24]
    mapped_sh[0,6,4] = sh[25]
    mapped_sh[0,6,5] = sh[26]
    mapped_sh[0,6,6] = sh[27]

    return mapped_sh

def map_dipy_to_pysh_o4(sh):
    mapped_sh = np.zeros((2, 5, 5))
    mapped_sh[0,0,0] = sh[0]
    mapped_sh[1,2,2] = sh[1]
    mapped_sh[1,2,1] = sh[2]
    mapped_sh[0,2,0] = sh[3]
    mapped_sh[0,2,1] = sh[4]
    mapped_sh[0,2,2] = sh[5]
    mapped_sh[1,4,4] = sh[6]
    mapped_sh[1,4,3] = sh[7]
    mapped_sh[1,4,2] = sh[8]
    mapped_sh[1,4,1] = sh[9]
    mapped_sh[0,4,0] = sh[10]
    mapped_sh[0,4,1] = sh[11]
    mapped_sh[0,4,2] = sh[12]
    mapped_sh[0,4,3] = sh[13]
    mapped_sh[0,4,4] = sh[14]

    return mapped_sh

rank1_rh_o4 = np.array([2.51327412, 1.43615664, 0.31914592])
rank1_rh_o6 = np.array([1.7951958 , 1.1967972 , 0.43519898, 0.06695369])
rank1_rh_o8 = np.array([1.3962634 , 1.01546429, 0.46867583, 0.12498022, 0.01470356])
