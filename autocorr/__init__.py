from ._version import get_versions


from .multitau import camera_multitau as multitau
from .fftautocorr import camera_fftautocorr as fftautocorr # noqa
from .timetime import camera_timetime as timetime

__all__ = [multitau, fftautocorr]
try:
    from .cAutocorr import camera_fftautocorr_mt as fftautocorr_mt # noqa
    from .cAutocorr import camera_multitau_mt as multitau_mt
    __all__ += [fftautocorr_mt, multitau_mt, timetime]
except ImportError:
    pass

__version__ = get_versions()['version']
del get_versions
