import sys

if sys.platform == "linux":
    # CUDA is loaded at runtime: the library opens libcublas.so.12 by name on first use,
    # and the cuBLAS version decides the exact floating point results. cuBLAS is loaded
    # here, in this order, so that the dlopen by name finds it:
    #  1. a libcublas.so.12 in a directory of LD_LIBRARY_PATH (an explicit choice);
    #  2. the nvidia-cublas-cu12 wheel (the "cuda" extra, or torch's CUDA 12 builds),
    #     which lives outside the library search path;
    #  3. whatever the system linker finds (a system CUDA installation).
    def _load_cublas():
        import ctypes
        import os

        mode = getattr(os, "RTLD_GLOBAL", 0)
        explicit = any(
            os.path.exists(os.path.join(d, "libcublas.so.12"))
            for d in os.environ.get("LD_LIBRARY_PATH", "").split(":")
            if d
        )
        if not explicit:
            try:
                import nvidia.cublas

                for base in nvidia.cublas.__path__:
                    lib_dir = os.path.join(base, "lib")
                    if os.path.exists(os.path.join(lib_dir, "libcublas.so.12")):
                        ctypes.CDLL(os.path.join(lib_dir, "libcublasLt.so.12"), mode=mode)
                        ctypes.CDLL(os.path.join(lib_dir, "libcublas.so.12"), mode=mode)
                        return
            except (ImportError, OSError):
                pass
        try:
            ctypes.CDLL("libcublas.so.12", mode=mode)
        except OSError:
            pass  # No cuBLAS: CUDA calls raise a clear error later.

    _load_cublas()
    del _load_cublas

try:
    from ctranslate2._ext import (
        DataType,
        Device,
        StorageView,
        WhisperDecoderState,
        contains_model,
        get_cuda_device_count,
        get_supported_compute_types,
        set_random_seed,
    )
    from ctranslate2.logging import get_log_level, set_log_level
except ImportError as e:
    # Allow using the Python package without the compiled extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from ctranslate2 import models
from ctranslate2.version import __version__, rapido_version

# converters and specs import torch (and, for converters, transformers) at module level.
# Those dependencies are only needed to convert models, not to run inference, so import
# these submodules on first use to keep "import ctranslate2" free of them.
_LAZY_SUBMODULES = ("converters", "specs")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))


# A wildcard import resolves ``__all__`` when it is defined and the module globals
# otherwise, so without this the lazy submodules would silently drop out of
# ``from ctranslate2 import *``. Deriving the list keeps the wildcard surface identical
# to what it was before they became lazy; a wildcard import asks for everything, so
# resolving them here is expected.
__all__ = sorted(
    [name for name in globals() if not name.startswith("_")] + list(_LAZY_SUBMODULES)
)
