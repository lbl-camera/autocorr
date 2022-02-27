import numpy as np

def camera_timetime(signal, algo='Sutton'):
    """Time-time of a signal
    Parameters
    ----------
    signal: 2-D or 3-D array
        input signal with time along slowest dimension e.g. [time, nrows, ncols]  or [time, npixels]
    algorithm: 'Brown' or 'Sutton'
        for details see:
            Brown: /* DOI:https://doi.org/10.1103/PhysRevE.56.6601 */
            Sutton: /* DOI: https://doi.org/10.1364/OE.11.002268 */
    Returns:
    --------
    time correlation function: numpy.ndarray
        should be self-explanatory
    """

    if len(signal.shape) == 2:
        A = signal.copy()
    elif len(signal.shape) > 3:
        nx, ny, nz = signal.shape
        A =  signal.reshape(nx, ny * nz)
    else:
        raise ValueError('Unsupported dimensions')

    if algo == 'Brown':
        from .cAutocorr import camera_twotime_brown_mt as tcf
    elif algo == 'Sutton':
        from .cAutocorr import camera_twotime_sutton_mt as tcf
    else:
        raise TypeError('Unknow algo')
    return tcf(signal)
