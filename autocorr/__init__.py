from ._version import get_versions


from .multitau import camera_multitau
from .fftautocorr import camera_fftautocorr  # noqa
from .twotime import camera_twotime

__all__ = [camera_multitau, camera_fftautocorr, camera_twotime]
try:
    from .cAutocorr import camera_fftautocorr_mt # noqa
    from .cAutocorr import camera_multitau_mt
    from .cAutocorr import camera_twotime_mt
    __all__ += [camera_fftautocorr_mt, camera_multitau_mt, camera_twotime_mt]
except ImportError:
    pass

__version__ = get_versions()['version']
del get_versions
