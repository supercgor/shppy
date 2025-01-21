import os
import sys
import numpy as np

from ctypes import c_float, c_void_p, c_long
from numpy.ctypeslib import load_library
from typing import Callable

_lib = load_library(f"libshppy", os.path.abspath(__file__))


def add_floats(x: float, y: float) -> float: ...

def square_array(arr: np.ndarray[float, np.dtype[np.float32]], n: int) -> None: ...


_lib.add_floats.argtypes = [c_float, c_float]
_lib.add_floats.restype = c_float

add_floats = _lib.add_floats

_lib.square_array.argtypes = [c_void_p, c_long]
_lib.square_array.restype = None

square_array = _lib.square_array

