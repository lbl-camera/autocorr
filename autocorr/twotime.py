import numpy as np


def camera_twotime(signal, lags_per_level=None):
    """Two-time correlation function of a signal.

    For details on two-time correlation function, please refer to O. Bikondoa, doi:10.1107/S1600576717000577

    Parameters
    ----------
    signal: 2-D array
        input signal, autocorrlation is calulated along `1-axis`
    lags_per_level: integer, optional
        number of lag-times per level, default is the full-signal length

    Returns:
    --------
    autocorrelations: numpy.ndarray
        should be self-explanatory
    lag-times: numpy.ndarray
        lag times in log2, corresponding to autocorrelation values
    """

    # if N is add subtract 1
    def even(x):
        return x if x % 2 == 0 else x - 1

    # 1-D data hack
    if len(signal.shape) == 1:
        N = even(signal.shape[0])
        a = np.array(signal[np.newaxis, :N].astype(np.float32), copy=True)
    elif len(signal.shape) == 2:
        # copy data a local array
        N = even(signal.shape[1])
        a = np.array(signal[:, :N], copy=True)
    elif len(signal.shape) > 2:
        raise ValueError('Flatten the [2,3,..] dimensions before passing to autocorrelate.')

    if lags_per_level is None:
        lags_per_level = N

    if N < lags_per_level:
        raise ValueError('Lag times per level must be greater than length of signal.')

    #  shorthand for long names
    m = lags_per_level

    # calculate levels
    levels = np.int_(np.log2(N / m)) + 1
    n2 = (levels + 1) * (m // 2)
    cvals = np.zeros((a.shape[0], n2), dtype=np.float32)

    # zero level
    cvals[:, :m] = a[:, :m]
    a = (a[:, :N:2] + a[:, 1:N:2]) / 2
    N = even(N // 2)

    for level in range(1, levels):
        n = np.arange(m // 2, m)
        idx = m // 2 + (level - 1) * (m // 2) + n
        cvals[:, idx] = a[:, m // 2:m]
        a = (a[:, :N:2] + a[:, 1:N:2]) / 2
        N = even(N // 2)
        if N < lags_per_level:
            break
    return np.mean(cvals[:, :, None] * cvals[:, None, :], axis=0)
