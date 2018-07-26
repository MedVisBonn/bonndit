import math

import numpy as np
import numpy.linalg as la

ix4 = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 1), (0, 0, 1, 2),
       (0, 0, 2, 2), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 2, 2), (0, 2, 2, 2),
       (1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2), (2, 2, 2, 2)]
mul4 = [1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1]


def encodeSym(W):
    return np.array([W[x] for x in ix4])


def decodeSym(W):
    r = np.zeros((3, 3, 3, 3))
    for i, x in enumerate(ix4):
        r[x] = W[i]
    return r


def make_rot_matrix4(L):
    r = np.zeros((15, 15))
    return r


# brute force \(^_^)/
def rotSym(W, L):
    Wm = decodeSym(W)

    Wwelle = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    s = [i, j, k, l]
                    s.sort()
                    for ii in range(3):
                        for jj in range(ii, 3):
                            for kk in range(jj, 3):
                                for ll in range(kk, 3):
                                    Wwelle[ii, jj, kk, ll] += L[i, ii] * L[
                                        j, jj] * L[k, kk] * L[l, ll] * Wm[
                                                                  s[0], s[1],
                                                                  s[2], s[3]]
    return encodeSym(Wwelle)


def power(v):
    return np.array([v[i[0]] * v[i[1]] * v[i[2]] * v[i[3]] for i in ix4])


def dot(a, b):
    r = 0
    for i in range(15):
        r += a[i] * b[i] * mul4[i]
    return r


def norm(a):
    return math.sqrt(dot(a, a))


def createPower(vv):
    T = np.zeros((15,))
    for v in vv:
        T = T + power(v)
    return T


TT = [[0, 1, 2, 3, 4, 5],
      [1, 3, 4, 6, 7, 8],
      [2, 4, 5, 7, 8, 9],
      [3, 6, 7, 10, 11, 12],
      [4, 7, 8, 11, 12, 13],
      [5, 8, 9, 12, 13, 14]]


def matrixify(T):
    M = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            M[a, b] = T[TT[a][b]]
    #	M = np.array([[T[0], T[1], T[2], T[3], T[4], T[5]],
    #	              [T[1], T[3], T[4], T[6], T[7], T[8]],
    #	              [T[2], T[4], T[5], T[7], T[8], T[9]],
    #	              [T[3], T[6], T[7], T[10], T[11], T[12]],
    #	              [T[4], T[7], T[8], T[11], T[12], T[13]],
    #	              [T[5], T[8], T[9], T[12], T[13], T[14]]])
    return M


def unmatrixify(M):
    T = np.zeros((15,))
    for i in range(15):
        n = 0
        for a in range(6):
            for b in range(6):
                if TT[a][b] == i:
                    T[i] += M[a, b]
                    n += 1
        T[i] /= n
    # print n

    return T


def ____matrixify2(T):
    M = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            M[a, b] = T[TT[a][b]] * mul4[TT[a][b]]
    return M


def ____unmatrixify2(M):
    T = np.zeros((15,))
    for i in range(15):
        n = 0
        for a in range(6):
            for b in range(6):
                if TT[a][b] == i:
                    T[i] += M[a, b]
                    n += 1
        T[i] /= n * mul4[i]
    # print n

    return T


TT9 = [[0, 1, 2, 1, 3, 4, 2, 4, 5],
       [1, 3, 4, 3, 6, 7, 4, 7, 8],
       [2, 4, 5, 4, 7, 8, 5, 8, 9],
       [1, 3, 4, 3, 6, 7, 4, 7, 8],
       [3, 6, 7, 6, 10, 11, 7, 11, 12],
       [4, 7, 8, 7, 11, 12, 8, 12, 13],
       [2, 4, 5, 4, 7, 8, 5, 8, 9],
       [4, 7, 8, 7, 11, 12, 8, 12, 13],
       [5, 8, 9, 8, 12, 13, 9, 13, 14]]


def matrixify9(T):
    M = np.zeros((9, 9))
    for a in range(9):
        for b in range(9):
            M[a, b] = T[TT9[a][b]]
    return M


def projectPosDef(M, verbose=False):
    if len(M.shape) != 2:
        raise Exception("M is not a matrix")
    n = M.shape[0]
    if M.shape[1] != n:
        raise Exception("M is not square")
    V, W = la.eigh(M)
    if verbose:
        print('EV: ', V)
    O = np.zeros(M.shape)
    for i, v in enumerate(V):
        if v > 1e-10:
            if verbose:
                print(i, v, W[i])
            for k in range(n):
                for l in range(n):
                    O[k, l] += v * W[k, i] * W[l, i]
    return O


def symmetrify(M):
    return matrixify(unmatrixify(M))


def iso(s):
    return [s, 0.0, 0.0, s / 3.0, 0.0, s / 3.0, 0.0, 0.0, 0.0, 0.0, s, 0.0,
            s / 3.0, 0.0, s]


def s_form(ten, v):
    return dot(ten, power(v))


def v_form(A, v):
    v000 = v[0] * v[0] * v[0]
    v001 = v[0] * v[0] * v[1]
    v002 = v[0] * v[0] * v[2]
    v011 = v[0] * v[1] * v[1]
    v012 = v[0] * v[1] * v[2]
    v022 = v[0] * v[2] * v[2]
    v111 = v[1] * v[1] * v[1]
    v112 = v[1] * v[1] * v[2]
    v122 = v[1] * v[2] * v[2]
    v222 = v[2] * v[2] * v[2]
    return np.array([A[0] * v000 + A[6] * v111 + A[9] * v222 + 6 * A[
        4] * v012 + 3 * (A[1] * v001 + A[2] * v002 + A[3] * v011 + A[
        5] * v022 + A[7] * v112 + A[8] * v122), \
                     A[1] * v000 + A[10] * v111 + A[13] * v222 + 6 * A[
                         7] * v012 + 3 * (
                         A[3] * v001 + A[4] * v002 + A[6] * v011 + A[
                         8] * v022 + A[11] * v112 + A[12] * v122), \
                     A[2] * v000 + A[11] * v111 + A[14] * v222 + 6 * A[
                         8] * v012 + 3 * (
                         A[4] * v001 + A[5] * v002 + A[7] * v011 + A[
                         9] * v022 + A[12] * v112 + A[13] * v122)])


def grad(A, v):
    res = np.array(v_form_f(res, A, v)) * 4.0
    proj = np.dot(res, v)
    return res - proj * np.array(v)


def mean(A):
    return 0.2 * (A[0] + A[10] + A[14] + 2 * (A[3] + A[5] + A[12]))


# def dist(A, B):
#	return math.sqrt(tensor.frobenius(A - B))

def eval4(T, v):
    return dot(T, power(v))


# def findLowestValue(T):
#	vmin = 10000000000
#	for i in range(1000):
#		w = np.random.rand(3)
#		w /= la.norm(w)
#		v = eval4(T, w)
#		vmin = min(v, vmin)
#	return vmin

def enhance(T, steps):
    M = matrixify(T)

    for i in range(steps):
        M = projectPosDef(M)
        M = symmetrify(M)
    return unmatrixify(M)
