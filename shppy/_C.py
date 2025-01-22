import os
import sys
import numpy as np

from ctypes import c_float, c_void_p, c_long
from numpy.ctypeslib import load_library
from typing import Callable

if sys.platform.startswith("win"):
    from ctypes import WinDLL
    _lib = WinDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libshppy.dll"))
else:
    from ctypes import CDLL
    if sys.platform.startswith("darwin"):
        _lib = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libshppy.dylib"))
    else:
        _lib = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libshppy.so"))

def add_floats(x: float, y: float) -> float: ...

def square_array(arr: np.ndarray[float, np.dtype[np.float32]], n: int) -> None: ...

_lib.add_floats.argtypes = [c_float, c_float]
_lib.add_floats.restype = c_float

add_floats = _lib.add_floats

_lib.square_array.argtypes = [c_void_p, c_long]
_lib.square_array.restype = None

square_array = _lib.square_array

