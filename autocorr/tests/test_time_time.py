import numpy as np
import autocorr
import time

N = 10240
np.random.seed(0)
t = np.arange(N)
A = np.exp(-0.05 * t)[:, np.newaxis] + np.random.rand(N, 24) * 0.1


def test_twotime_sutton():
    t0 = time.time()
    g2 = autocorr.timetime(A, algo='Sutton')
    t1 = time.time()
    print('sutton version = %f' % (t1 - t0))


def test_timetime_brown():
    t0 = time.time()
    g2 = autocorr.timetime(A, algo='Brown')
    t1 = time.time()
    print('brown version = %f' % (t1 - t0))
