import numpy as np
import numba as nb
import networkx as nx
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from typing import Sequence
from numpy.typing import NDArray

@nb.njit(fastmath=True)
def pbc_map(rs: np.ndarray, cell: np.ndarray | None = None, pbc: NDArray[np.bool_] | bool = False, align_center = False):
    if cell is None or not pbc:
        return rs
    
    if isinstance(pbc, bool):
        pbc = np.full(rs.shape[-1], pbc)
        
    cell = np.diag(cell) if cell.ndim == 1 else cell
    
    norm_rs = np.linalg.inv(cell.T) @ rs.T # (3, 3) @ (3, N) -> (3, N)
    for i in range(rs.shape[-1]):
        if pbc[i]:
            if align_center:
                norm_rs[i] -= np.rint(norm_rs[i])
            else:
                norm_rs[i] -= np.floor(norm_rs[i])
            
    return (cell.T @ norm_rs).T

@nb.njit(fastmath=True)
def pbc_repeat(rs: np.ndarray, cell: np.ndarray | None = None, pbc: NDArray[np.bool_] | bool = False):
    """
    Append periodic images of the positions to the input, use this when box_size is not diagonal, otherwise KDTree can be used directly.

    Args:
        rs (np.ndarray): _description_
        cell (np.ndarray): _description_
        pbc (Sequence[bool] | bool): _description_
    """
    if cell is None:
        return rs
    
    if isinstance(pbc, bool):
        if pbc:
            pbc = np.full(3, pbc)
        else:
            return rs
    
    N = 3 ** np.sum(pbc)
    
    out = np.empty((N * rs.shape[0], 3))
    out[:rs.shape[0]] = rs
    step = rs.shape[0]
    
    for i in range(3):
        if pbc[i]:
            np.subtract(out[:step], cell[i], out[step:step*2])
            np.add(out[:step], cell[i], out[step*2:step*3])
            step *= 3
    
    return out

def radius_query_kdtree(rs: NDArray[np.float_], k: int, cutoff: float, query: NDArray[np.float_] | None = None, cell: NDArray[np.float_] | None = None, pbc: NDArray[np.bool_] | bool = False):
    if query is None:
        query = rs

    if cell is None:
        tree = KDTree(rs)
        distances, bonds = tree.query(query, k=k, distance_upper_bound=cutoff, p=2)
        
    elif cell.ndim == 1:
        tree = KDTree(rs, boxsize=np.where(pbc, cell, 10000))
        distances, bonds = tree.query(query, k=k, distance_upper_bound=cutoff, p=2)
        
    elif cell.ndim == 2:
        N = len(rs)
        rs = pbc_repeat(rs, cell, pbc)
        tree = KDTree(rs)
        distances, bonds = tree.query(query, k=k, distance_upper_bound=cutoff, p=2)
        bonds = np.where(bonds < len(rs), bonds % N, N)
        
    else:
        raise ValueError(f"Cell should be in shape of [3,] or [3, 3], but got {cell.shape}.")

    return distances, bonds, tree

def radius_query_brute(rs: NDArray[np.float_], k: int, cutoff: float, query: NDArray[np.float_] | None = None, cell: NDArray[np.float_] | None = None, pbc: NDArray[np.bool_] | bool = False):
    if query is None:
        query = rs
    
    ds = rs[:, None] - query[None] # (N, 3) - (M, 3) -> (N, M, 3)
    ds = pbc_map(ds, cell, pbc)
    
    
    bonds = np.argpartition(ds, k, axis=1)[:, :k]
    distances = np.take_along_axis(ds, bonds, axis=1)
    mask = distances < cutoff
    distances = np.where(mask, distances, np.inf)
    bonds = np.where(mask, bonds, rs.shape[0])
    
    return distances, bonds, None
