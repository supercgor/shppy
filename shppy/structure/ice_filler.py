import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from ase import Atoms, io
from scipy.spatial import KDTree
from itertools import groupby, chain
from typing import Iterator, Any

# random.seed(1)

def map_pbc_vector(x: np.ndarray, box_size: np.ndarray | None = None, pbcs: list[bool] = [False, False, False]):
    if box_size is None:
        return x
    elif box_size.ndim == 1:
        box_size = np.diag(box_size)
    elif box_size.ndim == 2:
        pass
    else:
        raise ValueError("box_size should be a (3,) or (3, 3)")
    
    x = x @ np.linalg.inv(box_size)
    for i in range(3):
        if pbcs[i]:
            x[...,i] = x[...,i] - np.round(x[...,i])
    return x @ box_size

def _multigraph_eulerian_circuit(G, source):
    if G.is_directed():
        degree = G.out_degree
        edges = G.out_edges
    else:
        degree = G.degree
        edges = G.edges
    vertex_stack = [(source, None)]
    last_vertex = None
    last_key = None
    while vertex_stack:
        current_vertex, current_key = vertex_stack[-1]
        if degree(current_vertex) == 0:
            if last_vertex is not None:
                yield (last_vertex, current_vertex, last_key)
            last_vertex, last_key = current_vertex, current_key
            vertex_stack.pop()
        else:
            triple = random.choice(list(edges(current_vertex, keys=True)))
            _, next_vertex, next_key = triple
            vertex_stack.append((next_vertex, next_key))
            G.remove_edge(current_vertex, next_vertex, next_key)

def better_neighbour_finder(positions: np.ndarray,
                            cutoff: float,
                            k: int,
                            query: np.ndarray| None = None,
                            box_size: np.ndarray | None = None,
                            pbcs: list[bool] = [False, False, False],
                            ) -> tuple[np.ndarray, np.ndarray]:
    if query is None:
        query = positions
        
    if box_size is None:
        tree = KDTree(positions)
        distances, bonds = tree.query(query, k=k, distance_upper_bound=cutoff, p=2)
    elif box_size.ndim == 1:
        tree = KDTree(positions, boxsize=np.where(pbcs, box_size, 10000))
        distances, bonds = tree.query(query, k=k, distance_upper_bound=cutoff, p=2)
    elif box_size.ndim == 2:
        N = len(positions)
        query_positions = make_pbc_positions(positions, box_size, pbcs)
        tree = KDTree(query_positions)
        distances, bonds = tree.query(query, k=k, distance_upper_bound=cutoff, p=2)
        bonds = np.where(bonds < len(query_positions), bonds % N, N)
    else:
        raise ValueError("box_size should be a (3,) or (3, 3)")
    
    return distances, bonds

def check_anormaly_atoms(positions: np.ndarray,
                         G: nx.MultiGraph, 
                         idxs: list[int] | None = None,
                         tol: float = np.pi / 6,
                         cell: np.ndarray | None = None,
                         pbcs: list[bool] = [False, False, False],
                         ) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]:
    """
    Taking account of raw graph, check whether there are:
    - Oup atoms (degree = 3, has a bond pointing up) : (idx, neighbour)
    - Odown atoms (degree = 3, has a bond pointing down) : (idx, neighbour)
    - Oleft atoms (degree = 3, no bond pointing up or down) : idx
    
    Normally there is barely any Oup atoms. And there will be some Odown atoms if the surface does not cover the whole box.
    Oleft includes the upper layer of the surface and the lower layer of the bulk.
    
    Args:
        positions (np.ndarray): _description_
        G (nx.MultiDiGraph): _description_
        idxs (list[int] | None, optional): _description_. Defaults to None.
    """
    if idxs is None:
        idxs = G.nodes()
    
    Oup = set()
    Odown = set()
    Oleft = set()
    
    idxs = list(filter(lambda x: G.degree(x) < 4, idxs)) # type: ignore
    
    for idx in idxs:
        neighbours = list(G.neighbors(idx))
        uv = positions[neighbours] - positions[idx] # idx -> neighbours
        uv = map_pbc_vector(uv, cell, pbcs)
        uv = uv[:,2] / np.linalg.norm(uv, axis=1)
        
        mask = uv > np.cos(tol)
        
        if np.any(mask):
            Oup.add((idx, neighbours[np.nonzero(mask)[0].item()]))
            continue
        
        mask = uv < -np.cos(tol)
        
        if np.any(mask):
            s = neighbours[np.nonzero(mask)[0].item()]
            Odown.add((idx, s))
            continue
        
        Oleft.add(idx)
        
    return Oup, Odown, Oleft

def split_upper_lower(positions: np.ndarray,
                      G: nx.MultiGraph,
                      Oleft: set[int],
                      tol: float = np.pi / 6,
                      cell: np.ndarray | None = None,
                      pbcs: list[bool] = [False, False, False],
                      ):
    OHup, OHdown, OHleft = set(), set(), set()
    for idx in Oleft:
        neighbours = list(G.neighbors(idx))
        uv = positions[neighbours] - positions[idx]
        uv = map_pbc_vector(uv, cell, pbcs)
        uv = uv.sum(axis=0)
        uv = uv[2] / np.linalg.norm(uv)
        
        if uv > np.cos(tol):
            OHdown.add(idx)
        
        elif uv < -np.cos(tol):
            OHup.add(idx)
            
        else:
            OHleft.add(idx)
    
    return OHup, OHdown, OHleft
        

def find_hup(O_pos: np.ndarray,
             H_pos: np.ndarray,
             box_size: np.ndarray | None = None,
             pbcs: list[bool] = [False, False, False],
             OH_cutoff: float = 1.1,
             angle_cutoff: float = np.pi / 6,
             ) -> np.ndarray:
    
    # bonds: N x 2 : [0, 2N], N means no bond
    dd, bonds = better_neighbour_finder(H_pos, OH_cutoff, 2, O_pos, box_size, pbcs)
    
    query = np.concatenate([H_pos, [[0.0,0.0,-10000]]], axis=0)
    query = query[bonds]
    delta = query - O_pos[:,None,:]
    delta = map_pbc_vector(delta, box_size, pbcs)
    delta = delta[...,2] / np.linalg.norm(delta, axis=2) > np.cos(angle_cutoff)
    
    return np.any(delta, axis=1)

def make_pbc_positions(positions: np.ndarray, box_size: np.ndarray, pbcs: list[bool]):
    """
    Append periodic images of the positions to the input, use this when box_size is not diagonal, otherwise KDTree can be used directly.

    Args:
        positions (np.ndarray): _description_
        box_size (np.ndarray): _description_
        pbcs (list[bool]): _description_
    """
    for i in range(3):
        if pbcs[i]:
            positions = np.concatenate([positions, positions - box_size[i], positions + box_size[i]], axis=0)
    
    return positions

def generate_graph(positions: np.ndarray, 
                   box_size: np.ndarray | None = None, 
                   pbcs: list[bool] = [False, False, False], 
                   hup_idxs: list[int] | np.ndarray | None = None, 
                   detect_lower: bool = True, 
                   slab_thickness: float = 3.0,
                   max_bond_length: float = 3.4,
                   ):

    N = positions.shape[0]
    _, bonds = better_neighbour_finder(positions, max_bond_length, 5, None, box_size, pbcs)
    
    G = nx.MultiGraph()
    for i, pos in enumerate(positions):
        G.add_node(i, position=pos, vitual = False)
    
    for i, js in enumerate(bonds):
        for j in js:
            if i < j and j < N:
                G.add_edge(i, j, auto=False)
    
    Oup, Odown, Oleft = check_anormaly_atoms(positions, G, None, np.pi / 6, box_size, pbcs)
    
    OHup, OHdown, OHleft = split_upper_lower(positions, G, Oleft, np.pi / 6, box_size, pbcs)
    
    G.add_node(N,   position=np.array([0, 0, 10000]),  virtual=True)    # directed nodes for Hup specifying.
    G.add_node(N+1, position=np.array([0, 0, -10000]), virtual=True) # undirected nodes for any nodes that is not 4-degree.
    
    for idx in OHdown:
        G.add_edge(idx, N + 1, auto=False)
    
    for idx in OHup:
        G.add_edge(idx, N, auto=False)

    N = len(positions)
    print(Oup, Odown)
    for idx in chain(map(lambda x: x[0], Oup), map(lambda x: x[0], Odown)):
        G.add_edge(idx, N + 1, auto=True)
    
    start = next(filter(lambda x: G.degree(x) % 2, G.nodes()), random.randint(0, N-1))
    
    init_path = list(_multigraph_eulerian_circuit(G.copy(), start))
    
    if hup_idxs is None:
        return list(filter(lambda x: x[0] < N, init_path)), G, {"Oup": Oup, "Odown": Odown, "Oleft": Oleft, "OHup": OHup, "OHdown": OHdown, "OHleft": OHleft}
    
    H = nx.MultiDiGraph()
    for i in init_path:
        H.add_edge(*i, auto=G.get_edge_data(*i)["auto"])
    
    H_hup = set(H.predecessors(N))
    H_hdown = set(H.successors(N))
    
    poor_hup, poor_hdown = set(), set()
    hup_idxs = set(hup_idxs)
    
    for i in H_hup:
        if i not in hup_idxs:
            poor_hup.add(i)
        H.remove_edge(i, N)
        
    for i in H_hdown:
        if i in hup_idxs:
            poor_hdown.add(i)   
        H.remove_edge(N, i)
            
    def reverse_edge(G, i, j, k = None):
        if k is None:
            k = next(iter(G.get_edge_data(i, j).keys()))
        edge_data = G.get_edge_data(i, j, k)
        G.remove_edge(i, j, k)
        G.add_edge(j, i, k, **edge_data)
    
    tmp = set()
    
    for i_down in poor_hdown:
        if len(poor_hup):
            path = next(nx.all_simple_edge_paths(H, i_down, poor_hup))
            i_up = path[-1][1]
            for i, j, k in path:
                reverse_edge(H, i, j, k)
            poor_hup.remove(i_up)
            tmp.add(i_up)
        else:
            path = next(nx.all_simple_edge_paths(H, i_down, N))
            for i, j, k in path:
                reverse_edge(H, i, j, k)
            reverse_edge(H, N, N+1)
    
    while len(poor_hup):
        path = next(nx.all_simple_edge_paths(H, N+1, poor_hup))
        for i, j, k in path:
            reverse_edge(H, i, j, k)
        poor_hup.remove(path[-1][1])
        tmp.add(path[-1][1])
    
    for idx in H_hup:
        if idx in tmp:
            H.add_edge(N, idx, auto=False)
        else:
            if idx not in hup_idxs:
                print(idx)
            H.add_edge(idx, N, auto=False)

    for idx in H_hdown:
        if idx in poor_hdown:
            H.add_edge(idx, N, auto=False)
        else:
            H.add_edge(N, idx, auto=False)
    
    for _ in range(H.in_degree(N) - H.out_degree(N)):
        H.add_edge(N, N+1, auto=False)
    for _ in range(H.out_degree(N) - H.in_degree(N)):
        H.add_edge(N+1, N, auto=False)
    
    
    for i in sorted(H.nodes()):
        print(i, H.in_degree(i), H.out_degree(i))
        
    start = next(filter(lambda x: H.out_degree(x) % 2, H.nodes()), random.randint(0, N-1))
    new_path = list(_multigraph_eulerian_circuit(H.reverse(), start))
    
    """while True:
        try:
            n = next(G.predecessors(N))
            # shortest_ = nx.shortest_path_length(G, N, n)
            # path = next(nx.all_simple_paths(G, N, n, cutoff=shortest_ + 0))
            path: list = nx.bidirectional_shortest_path(G, N, next(G.predecessors(N))) # type: ignore
            path_edges = []
            
            # (N -> idx)
            edges = G.get_edge_data(path[0], path[1])
            
            k: int = next(iter(edges.keys()))
            edge_attr = edges[k]
            
            G.remove_edge(path[0], path[1], k)
            path_edges.append(((path[0], path[1],k), edge_attr))
            # (idx <-> ... <-> jdx)
            
            for i, j in zip(path[1:-1], path[2:]):
                edge_attr = G.get_edge_data(i, j, 0)
                G.remove_edge(i, j, 0)
                G.remove_edge(j, i, 0)
                path_edges.append(((i, j, 0), edge_attr))
            
            # (jdx -> N)
            edges = G.get_edge_data(path[-1], N)
            
            k = next(iter(edges.keys()))
            edge_attr = edges[k]
            
            G.remove_edge(path[-1], N, k)
            path_edges.append(((path[-1], N, k), edge_attr))
            closes_loops.append(path_edges)
            
        except StopIteration:
            break
    """
    
    return list(filter(lambda x: x[0] < N, new_path)), H, {"Oup": Oup, "Odown": Odown, "Oleft": Oleft, "OHup": OHup, "OHdown": OHdown, "OHleft": OHleft}
    
if __name__ == "__main__":
    f = "/Users/supercgor/Documents/data/exp/solution/raw/1h.poscar"
    # f = "/Users/supercgor/Documents/data/exp/solution/raw/558.poscar"
    atoms: Atoms = io.read(f, format="vasp")
    box_size = atoms.cell.array
    pbcs = [True, True, False]
    O_atoms: Atoms = atoms[atoms.get_atomic_numbers() == 8]
    H_atoms: Atoms = atoms[atoms.get_atomic_numbers() == 1]
    
    O_pos = O_atoms.positions
    H_pos = H_atoms.positions
    print(len(O_atoms))
    
    zmax = np.max(O_atoms.positions[:,2]) - 2.0

    mask = (O_atoms.positions[:,2] > zmax).nonzero()[0]
    
    # Hup_idxs = find_hup(O_atoms.positions, H_atoms.positions, box_size, atoms.get_pbc())    
    Hup_mask = find_hup(O_pos[mask], H_pos, box_size, pbcs)
    
    middle = np.zeros(O_pos.shape[0])
    middle[mask] = Hup_mask
    Hup_idxs = middle.nonzero()[0]
    print(Hup_idxs)
    
    results, G, dic = generate_graph(O_atoms.positions, box_size, pbcs, Hup_idxs)
    # print(len(results))
    
    # for i in results:
    #     print(i, G.get_edge_data(*i))
    
    non_auto, auto = [], []
    
    for x in results:
        if G.get_edge_data(*x)["auto"]:
            auto.append(x)
        else:
            non_auto.append(x)
    
    non_auto = np.asarray(non_auto)
    auto = np.asarray(auto)
    query_pos = np.concatenate([O_atoms.positions, 
                                [[0.0, 0.0,  10000], 
                                 [0.0, 0.0, -10000]]], axis=0)
    
    
    selected_pos = query_pos[non_auto[:,:2]]
    
    selected_pos, diff = selected_pos[:, 0], selected_pos[:,1] - selected_pos[:,0]
    
    diff = map_pbc_vector(diff, box_size, [True, True, False])
    diff_norm = np.linalg.norm(diff, axis=1, keepdims=True)
    diff = diff / diff_norm

    selected_pos = selected_pos + diff
    
    if len(auto) != 0:
        auto_pos = query_pos[auto[:,0]].copy()
        
        for i, (idx, _, _) in enumerate(auto):
            if isinstance(G, nx.MultiDiGraph):
                nei = list(filter(lambda x: x < len(O_atoms), chain(G.successors(idx), G.predecessors(idx))))
            else:
                nei = list(filter(lambda x: x < len(O_atoms), chain(G.neighbors(idx))))
            # nei.remove(len(O_atoms) + 1)
            neighbors_pos = query_pos[nei]
            uv = neighbors_pos - query_pos[idx]
            uv = map_pbc_vector(uv, box_size, pbcs)
            uv = - uv.sum(axis=0)
            uv = uv / np.linalg.norm(uv)
            auto_pos[i] += uv
        
        selected_pos = np.concatenate([selected_pos, auto_pos], axis=0)
    H_atoms = Atoms("H" * len(selected_pos), positions=selected_pos)
    
    O_atoms.set_array("idx", np.arange(len(O_atoms)))
    
    for k, inxs in dic.items():
        arr = np.zeros(len(O_atoms))
        if k in ["Oup", "Odown"]:
            inxs = map(lambda x: x[0], inxs)
        arr[list(inxs)] = 1
        O_atoms.set_array(k, arr)
    H_atoms.set_array("idx", np.arange(len(O_atoms), len(H_atoms) + len(O_atoms)))
    
    atoms = O_atoms + H_atoms
    
    print(len(atoms))
    io.write("tools/test.xyz", atoms, format="extxyz")