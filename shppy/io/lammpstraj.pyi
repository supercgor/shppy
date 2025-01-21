import numpy as np
from typing import overload

from ..atom import Atoms
from .._types import PathLike

def read_timestep(path: PathLike) -> np.ndarray: ...

@overload
def read_lammps_dump_text(path: PathLike, index: int, sort_by_id: bool = True, type_map: dict[int, int] | None = None) -> Atoms: ...

@overload
def read_lammps_dump_text(path: PathLike, index: slice, sort_by_id: bool = True, type_map: dict[int, int] | None = None, num_workers: int = 1) -> list[Atoms]: ...
