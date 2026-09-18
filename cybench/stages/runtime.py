"""Runtime initialization shared by GPU-backed stage entry points."""

from __future__ import annotations

import os
import sys


_DLL_HANDLES = []


def initialize_torch_before_data_libraries():
    """Initialize CUDA PyTorch before Pandas on Windows.

    Some Conda environments expose incompatible DLL search paths after NumPy or
    Pandas is imported. Keeping the added-directory handles alive and importing
    Torch first prevents c10.dll initialization failures in Qwen and TabPFN.
    """
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        candidates = (
            os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
            os.path.join(sys.prefix, "Library", "bin"),
            os.path.join(sys.prefix, "Library", "mingw-w64", "bin"),
        )
        for path in candidates:
            if os.path.isdir(path):
                _DLL_HANDLES.append(os.add_dll_directory(path))
    import torch

    return torch
