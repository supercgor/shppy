import ase.atom
import numpy as np
import ase
                 
from typing import overload, Annotated, Literal, Union
from numpy.typing import NDArray
np.ndarray
class Atoms(ase.Atoms):
    @property
    def symbols(self) -> str: ...

    @property
    def numbers(self) -> NDArray[np.int_]: ...

    @property
    def positions(self) -> NDArray[np.float_]: ...

    @overload
    def __getitem__(self, key: int) -> ase.atom.Atom: ...
    
    @overload
    def __getitem__(self, key: np.int_) -> ase.atom.Atom: ...
    
    @overload
    def __getitem__(self, key: slice | tuple | list) -> ase.Atoms: ...
    
    @overload
    def __getitem__(self, key: Annotated[NDArray[np.bool_], Literal["N"]]) -> ase.Atoms: ...
    
    @overload
    def __getitem__(self, key: Annotated[NDArray[np.int_], Literal["N"]]) -> ase.Atoms: ...
    
    @property
    def cell(self) -> np.ndarray: ...
    
    def get_cell(self, complete = False) -> np.ndarray: ...
    
    @property
    def xs(self) -> np.ndarray: ...
    
    @property
    def ys(self) -> np.ndarray: ...
    
    @property
    def zs(self) -> np.ndarray: ...
