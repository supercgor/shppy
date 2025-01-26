from pathlib import Path
from ase.io import (read as read_ase, write)
from typing import overload

from .lammpstraj import read_lammps_dump_text, read_timestep
from .._types import PathLike, AtomFormat
from ..atom import Atoms

@overload
def read(filename: PathLike, index: slice = slice(-1), format: AtomFormat | None = None, parallel: bool = False, do_not_split_by_at_sign: bool = False, **kwargs) -> list[Atoms]: ...

@overload
def read(filename: PathLike, index: int, format: AtomFormat | None = None, parallel: bool = False, do_not_split_by_at_sign: bool = False, **kwargs) -> Atoms: ...


def read(filename, index = slice(-1), format = None, parallel = False, do_not_split_by_at_sign = False, **kwargs): # type: ignore
    filename = Path(filename)
    if filename.suffix in [".lammpstrj", ".pos"] or format == "lammps-dump-text":
        return read_lammps_dump_text(filename, index, num_workers= -1 if parallel else 1, **kwargs)
    else:
        out = read_ase(filename, index, format, parallel, do_not_split_by_at_sign, **kwargs)
        if isinstance(index, slice) and isinstance(out, Atoms):
            return [out]
        else:
            return out
