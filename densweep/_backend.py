"""
densweep._backend
─────────────────
Tiny array-backend shim so the from-scratch HDBSCAN can run the heavy
O(n^2) numerics on either NumPy (CPU) or CuPy (GPU) with the same code.

We never *require* CuPy. ``get_xp("gpu")`` returns CuPy if it is importable
and a device is visible, otherwise it transparently falls back to NumPy and
remembers to warn only once.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

_GPU_WARNED = False
_CUPY = None
_CUPY_CHECKED = False


def _try_import_cupy():
    global _CUPY, _CUPY_CHECKED
    if _CUPY_CHECKED:
        return _CUPY
    _CUPY_CHECKED = True
    try:
        import cupy as cp  # type: ignore

        # Touch the runtime so a driver-less install fails here, not later.
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device visible")
        _CUPY = cp
    except Exception:
        _CUPY = None
    return _CUPY


def gpu_available() -> bool:
    """True if CuPy + a usable CUDA device are present."""
    return _try_import_cupy() is not None


def get_xp(device: str = "cpu"):
    """Return the array module (``numpy`` or ``cupy``) for ``device``.

    ``device="gpu"`` falls back to NumPy (with a one-time warning) when CuPy
    or a CUDA device is unavailable, so library code stays portable.
    """
    global _GPU_WARNED
    if device == "gpu":
        cp = _try_import_cupy()
        if cp is not None:
            return cp
        if not _GPU_WARNED:
            warnings.warn(
                "densweep: GPU requested but CuPy/CUDA is unavailable; "
                "falling back to the NumPy CPU backend.",
                RuntimeWarning,
                stacklevel=2,
            )
            _GPU_WARNED = True
    return np


def to_numpy(a: Any) -> np.ndarray:
    """Bring an array back to the host as a contiguous NumPy array."""
    if a is None:
        return None
    cp = _CUPY
    if cp is not None and isinstance(a, cp.ndarray):  # pragma: no cover - GPU only
        return cp.asnumpy(a)
    return np.asarray(a)


# Public alias used by gpu_check-style diagnostics in the original project.
_to_numpy = to_numpy
