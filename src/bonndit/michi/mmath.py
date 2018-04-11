#!/usr/bin/python
# -*- coding: utf-8 -*-


# hmmm, python2 doesn't like this
# from math import factorial
import numpy as np


def factorial(n):
    if n < 16:
        return [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200,
                1307674368000][n]
    r = 1307674368000
    for i in range(16, n + 1):
        r *= i
    return r


def binom(n, k):
    return int(factorial(n) / factorial(k) / factorial(n - k))


def multinom(i, j, k):
    return int(factorial(i + j + k) / factorial(i) / factorial(j) / factorial(k))


# helper for cos(n*x)
#   returns an array of coefficients of
#      cos(n*x) = sum_i C_i * cos^{n-i}(x)*sin^i(x)
def cos_n(n):
    r = np.zeros(n + 1)
    for i in range(n // 2 + 1):
        r[i * 2] = (-1) ** (i) * binom(n, 2 * i)
    return r


# sin(n*x)
def sin_n(n):
    r = np.zeros(n + 1)
    for i in range((n + 1) // 2):
        r[i * 2 + 1] = (-1) ** (i) * binom(n, 2 * i + 1)
    return r
