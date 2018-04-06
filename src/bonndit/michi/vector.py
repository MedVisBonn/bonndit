import numpy as np
import numpy.linalg as la
import math

v_0 = np.array((0, 0, 0))
e_x = np.array((1, 0, 0))
e_y = np.array((0, 1, 0))
e_z = np.array((0, 0, 1))


def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm_sqr(v):
    return v[0] ** 2 + v[1] ** 2 + v[2] ** 2


def norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def normalized(v):
    l = norm(v)
    if l > 0:
        return v / l
    return e_z


# project v onto direction
def project(v, direction):
    d = normalized(direction)
    return v - dot(v, d) * d


# find an ortho normal basis including a
def ortho_normal(a):
    if abs(a[2]) < 0.8:
        b = cross(a, e_z)
    else:
        b = cross(a, e_x)
    b = b / norm(b)
    c = cross(a, b)
    return b, c


# matrix M for   M*w = v x w
def cross_matrix(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# assuming |n| = 1
def rotation(n, phi):
    nn = np.outer(n, n)
    e = np.eye(3)
    return nn + math.cos(phi) * (e - nn) - math.sin(phi) * cross_matrix(n)


def rotation_x(phi):
    return np.array([[1, 0, 0], [0, math.cos(phi), math.sin(phi)], [0, -math.sin(phi), math.cos(phi)]])


def rotation_y(phi):
    return np.array([[math.cos(phi), 0, -math.sin(phi)], [0, 1, 0], [math.sin(phi), 0, math.cos(phi)]])


def rotation_z(phi):
    return np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])


def rotation_xyz(phi):
    mx = rotation_x(phi[0])
    my = rotation_y(phi[1])
    mz = rotation_z(phi[2])
    return np.dot(mz, np.dot(my, mx))


# create a rotation matrix R, so that R*a = b
# assume a,b normalized
def rotation_onto(a, b):
    d = dot(a, b)
    if d > 0.999999:
        # almost a = b   => identity
        return np.eye(3)
    if d < -0.999999:
        # almost a = -b
        axis, _ = ortho_normal(a)
    else:
        axis = normalized(cross(a, b))
    return rotation(axis, -math.acos(d))


# v: 1 vector (phi,theta)
#    or list of vectors [N,2]
def sphere_to_cart(v):
    v = np.array(v)
    if len(v.shape) == 1:
        ct = math.cos(v[0])
        st = math.sin(v[0])
        cp = math.cos(v[1])
        sp = math.sin(v[1])
        return np.array([st * cp, st * sp, ct])
    elif len(v.shape) == 2:
        ct = np.cos(v[:, 0])
        st = np.sin(v[:, 0])
        cp = np.cos(v[:, 1])
        sp = np.sin(v[:, 1])
        return np.array([st * cp, st * sp, ct]).T


# v: 1 vector
#        returns (r,phi,theta)
#    or list of vectors [N,3]
#        returns [N,3]
def cart_to_sphere(v):
    v = np.array(v)
    if len(v.shape) == 1:
        r = norm(v)
        if r == 0:
            return v_0
        phi = math.atan2(v[1], v[0])
        theta = math.atan2(math.sqrt(v[0] * v[0] + v[1] * v[1]), v[2])
        return np.array([r, theta, phi])
    elif len(v.shape) == 2:
        r = la.norm(v, axis=1)
        phi = np.arctan2(v[:, 1], v[:, 0])
        theta = np.arctan2(np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]), v[:, 2])
        return np.stack([r, theta, phi]).T
