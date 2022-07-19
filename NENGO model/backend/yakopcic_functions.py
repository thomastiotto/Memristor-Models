import numpy as np
from scipy.stats import truncnorm


def mimd(v, g_p, b_p, g_n, b_n):
    return np.where(v >= 0, g_p * np.sinh(b_p * v), g_n * np.sinh(b_n * v))


def mim_iv(v, g, b):
    return g * np.sinh(b * v)


def h1(v, g_p, b_p, g_n, b_n):
    return np.where(v >= 0, g_p * np.sinh(b_p * v), g_n * (1 - np.exp(-b_n * v)))


def h2(v, g_p, b_p, g_n, b_n):
    return np.where(v >= 0, g_p * (1 - np.exp(-b_p * v)), g_n * np.sinh(b_n * v))


#def current(v, x, gmax_p, bmax_p, gmax_n, bmax_n, gmin_p, bmin_p, gmin_n, bmin_n):  # First implementation
#    return mimd(v, gmax_p, bmax_p, gmax_n, bmax_n) * x + mimd(v, gmin_p, bmin_p, gmin_n, bmin_n) * (1 - x)


def current(v, x, gmax_p, bmax_p, gmax_n, bmax_n, gmin_p, bmin_p, gmin_n, bmin_n): # Implemented with Dima (2022)
    return h1(v, gmax_p, bmax_p, gmax_n, bmax_n) * x + h2(v, gmin_p, bmin_p, gmin_n, bmin_n) * (1 - x)


def g(v, Ap, An, Vp, Vn):
    return np.select([v > Vp, v < -Vn], [Ap * (np.exp(v) - np.exp(Vp)), -An * (np.exp(-v) - np.exp(Vn))], default=0)


def wp(x, xp):
    return (xp - x) / (1 - xp) + 1


def wn(x, xn):
    return x / (1 - xn)


def f(v, x, xp, xn, alphap, alphan, eta):
    return np.select([eta * v >= 0, eta * v <= 0],
                     [np.select([x >= xp, x < xp],
                                [np.exp(-alphap * (x - xp)) * wp(x, xp),
                                 1]),
                      np.select([x <= xn, x > xn],
                                [np.exp(alphan * (x + xn)) * wn(x, xn),
                                 1])
                      ])


def dxdt(v, x, Ap, An, Vp, Vn, xp, xn, alphap, alphan, eta=None):
    eta = 1 if eta is None else eta
    # print("g: ",  g(v, Ap, An, Vp, Vn))
    # print("f: ", f(v, x, xp, xn, alphap, alphan, eta))
    return eta * g(v, Ap, An, Vp, Vn) * f(v, x, xp, xn, alphap, alphan, eta)


def get_truncated_normal(mean, sd, low, upp, out_size, in_size):
    try:
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd) \
            .rvs(out_size * in_size) \
            .reshape((out_size, in_size))
    except ZeroDivisionError:
        return np.full((out_size, in_size), mean)


def resistance2conductance(R, r_min, r_max):
    g_min = 1.0 / r_max
    g_max = 1.0 / r_min
    g_curr = 1.0 / R

    g_norm = (g_curr - g_min) / (g_max - g_min)

    return g_norm
