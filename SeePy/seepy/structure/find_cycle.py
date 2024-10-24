import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from ase import Atoms, io
from collections import defaultdict
from functools import partial
from scipy.spatial import KDTree
from typing import Iterator

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

def is_even_crossing(cycle: tuple[int], positions: np.ndarray, cell:np.ndarray, pbcs: list[bool]):
    cycles_pos = positions[list(cycle)]
    if cell.ndim == 1:
        cell = np.diag(cell)
    
    cycles_pos = np.abs(cycles_pos - np.roll(cycles_pos, 1, axis=0)) 
    cycles_pos = np.dot(cycles_pos, np.linalg.inv(cell))
    counts = np.sum(cycles_pos[:,pbcs] > 0.5)
    return counts != 1

    

def any_sub_cycle(indices: np.ndarray, cycle: tuple):
    """
    _summary_

    Args:
        bonds (EdgeView): _description_
        cycle (tuple): (0, 1, 3, 5, 7, ...).

    Returns:
        _type_: _description_
    """
    eles = len(cycle)
    
    if eles <= 3:
        return False
    elif eles == 4:
        return (cycle[2] in indices[cycle[0]]) or (cycle[1] in indices[cycle[3]])
    else:
        for i, idx in enumerate(cycle):
            for j, jdx in enumerate(cycle):
                delta = i - j
                delta -= len(cycle) * np.rint(delta / len(cycle))
                if delta > 1 and idx in indices[jdx]:
                    return True
        return False
    
def sort_cycle(nodes):
    min_ele = nodes.index(min(nodes))
    n = len(nodes)
    if nodes[(min_ele + 1) % n] < nodes[(min_ele - 1) % n]:
        out = nodes[min_ele:] + nodes[:min_ele]
    else:
        out = nodes[min_ele::-1] + nodes[:min_ele:-1]
    return out

def find_cycle(positions: np.ndarray,
               box_size: np.ndarray | None = None, 
               pbcs: list[bool] = [False, False, False],
               indices: list[int] | None = None, 
               bond_length: float = 3.6, 
               max_ring_diameter: float = 7.0, 
               max_ring_elements: int = 8,
               anomaly_check: bool = True,
               ) -> tuple[set[tuple[int]], list[tuple[int]], set[int], set[int] | None]:
    """
    Find cycles for a given set of atom positions.
 
    Args:
        positions           (np.ndarray): (N, 3) array of atom positions.
        box_size            (np.ndarray, optional): (3,) or (3, 3) array, periodic boundary conditions.
        pbcs                (list[bool], optional): Periodic boundary conditions. Defaults to [False, False, False].
        indices             (list[int], optional): List of atom indices to consider, useful when non-rectangular boundary is considered. Defaults to None.
        bond_length         (float, optional): The bond length between two atoms, Dont set it too large, some unexpected results may occur. Defaults to 3.6.
        max_ring_diameter   (float, optional): Defaults to 7.0.
        max_ring_elements   (int, optional): Defaults to 8.
        anomaly_check       (bool, optional): Check if there are any anomaly atoms or edges, if True, the detection will run twice, and cancel out the anomaly atoms and edges. Defaults to True.
    Returns:
        tuple:
        - cycles              (set[tuple[int]]): A set of cycles, each cycle is a tuple of atom indices.
        - boundaries          (list[tuple[int]]): A list of boundaries, each boundary is a tuple of atom indices.
        - isolated_nodes      (set[int]): A set of isolated nodes.
        - anomaly_atoms       (set[int]): A set of anomaly atoms.
    """
    N = positions.shape[0]
    if box_size is None:
        pbc_box = None
        tree = KDTree(positions, boxsize=None)
        bonds_d, bonds = tree.query(positions, k=5, distance_upper_bound=bond_length, p=2)
        adj = tree.query_ball_point(positions, max_ring_diameter, p=2)
        
    elif box_size.ndim == 1:
        if not any(pbcs):
            pbc_box = None
        else:
            pbc_box = box_size.copy()
            for i in range(3):
                if not pbcs[i]:
                    pbc_box[i] *= 10000
        
        tree = KDTree(positions, boxsize=pbc_box)
        bonds_d, bonds = tree.query(positions, k=5, distance_upper_bound=bond_length, p=2)
        adj = tree.query_ball_point(positions, max_ring_diameter, p=2)
        
    elif box_size.ndim == 2:
        if np.allclose(box_size, np.diag(np.diagonal(box_size))):
            box_size = np.diagonal(box_size)
            return find_cycle(positions, box_size, pbcs, indices, bond_length, max_ring_diameter, max_ring_elements)
        
        query_positions = positions.copy()
        for i in range(3):
            if pbcs[i]:
                query_positions = np.concatenate((query_positions, query_positions - box_size[i], query_positions + box_size[i]))
        
        tree = KDTree(query_positions, boxsize=None)
        bonds_d, bonds = tree.query(positions, k=5, distance_upper_bound=bond_length, p=2)
        adj = tree.query_ball_point(positions, max_ring_diameter, p=2)
        
        bonds = np.where(bonds < len(query_positions), bonds % N, N)
        adj = list(map(lambda x: list(map(lambda y: y % N if y < N else N, x)), adj))
        
    else:
        raise ValueError("box_size should be a (3,) or (3, 3)")
    
    if box_size is not None:
        if box_size.ndim == 1:
            need_check_crossing = np.any(box_size < 2 * (max_ring_diameter + bond_length))
        elif box_size.ndim == 2:
            need_check_crossing = np.any(np.linalg.norm(box_size, axis=1)  < 2 * (max_ring_diameter + bond_length))
        else:
            raise ValueError("box_size should be a (3,) or (3, 3)")
        check_func = partial(is_even_crossing, positions = positions, cell=box_size, pbcs=pbcs)
    else:
        need_check_crossing = False
        check_func = None
    
    G = nx.Graph()
    
    # Add nodes to the graph, with `position` attribute
    for i, pos in enumerate(positions):
        G.add_node(i, position=pos)
    
    # Add edges to the graph, with bond length `r` and number of counts `n`.
    for idx, js in enumerate(bonds):
        for j, jdx in enumerate(js):
            if jdx < N and idx != jdx:
                G.add_edge(idx, jdx, r=bonds_d[idx, j], n=0)

    if anomaly_check:
        H = G.copy()

    cycles = bfs_basic_cycles(G, bonds, adj, max_ring_elements, indices, need_check_crossing, check_func = check_func)
    
    boundaries = list()
    boundaries_nodes = set()
    isolated_nodes = set()
    boundaries_all = nx.connected_components(G)
    
    for nodes in boundaries_all:
        if len(nodes) <= 3:
            continue
        subgraph = G.subgraph(nodes)
        for boundaries_cycle in nx.simple_cycles(subgraph):
            boundaries.append(sort_cycle(tuple(boundaries_cycle)))
            boundaries_nodes.update(set(boundaries_cycle))
        # print(boundaries_nodes)
        for i in nodes:
            if i not in boundaries_nodes:
                isolated_nodes.add(i)
                
    if anomaly_check:
        anomaly_counts_atoms = defaultdict(int)
        anomaly_counts_edges = defaultdict(int)
        for cycle in cycles:
            if len(cycle) <= 4:
                for idx in cycle:
                    anomaly_counts_atoms[idx] += 1
                for c_i, c_j in zip(cycle, cycle[1:] + (cycle[0],)):
                    c_i, c_j = min(c_i, c_j), max(c_i, c_j)
                    anomaly_counts_edges[(c_i, c_j)] += 1
        
        anomaly_atoms = set([k for k, v in anomaly_counts_atoms.items() if v >= 3])
        anomaly_edges = [(k, v) for (k, v), c in anomaly_counts_edges.items() if c >= 2]
                
        print(f"Anomaly atoms: {anomaly_atoms}")
        print(f"Anomaly edges: {anomaly_edges}")
        
        if len(anomaly_atoms) == 0 and len(anomaly_edges) == 0:
            return cycles, boundaries, isolated_nodes, None
        
        for (c_i, c_j) in anomaly_edges:
            H.remove_edge(c_i, c_j)
        
        for i in anomaly_atoms:
            H.remove_node(i)
        
        for (k, v) in anomaly_edges:
            bonds[k, np.where(bonds[k] == v)] = N
            bonds[v, np.where(bonds[v] == k)] = N
        
        cycles = bfs_basic_cycles(H, bonds, adj, max_ring_elements, indices, need_check_crossing, check_func = check_func)
    
        boundaries = list()
        boundaries_nodes = set()
        isolated_nodes = set()
        boundaries_all = nx.connected_components(H)
        
        for nodes in boundaries_all:
            if len(nodes) <= 3:
                continue
            subgraph = H.subgraph(nodes)
            for boundaries_cycle in nx.simple_cycles(subgraph):
                boundaries.append(sort_cycle(tuple(boundaries_cycle)))
                boundaries_nodes.update(set(boundaries_cycle))
            # print(boundaries_nodes)
            for i in nodes:
                if i not in boundaries_nodes:
                    isolated_nodes.add(i)
        
        return cycles, boundaries, isolated_nodes, anomaly_atoms
    
    else:
        return cycles, boundaries, isolated_nodes, None

def bfs_basic_cycles(G: nx.Graph, 
                     bonds: np.ndarray, 
                     adj: list[list[int]], 
                     max_ring_elements: int, 
                     indices: list[int] | None = None, 
                     need_check: bool | np.bool_ = False,
                     check_func: callable = None
                     ):
    total_cycles = set()
    if need_check:
        assert check_func is not None, "check_func should be provided when need_check is True"
    
    if indices is None:
        iter = G
    else:
        iter = indices
    for root in iter:
        added = 0
        queue = [root]
        layers = {root: 0}
        branch = {root: [root]}
 
        def find_next_layer(nbr, layer):
            nonlocal added
            this_branch = branch[nbr]
            for i in list(sorted(G[nbr], key=lambda x: G.degree[x])): # type: ignore
                if i not in adj[i] or i == root:
                    continue
                elif i not in branch or layer == 0:
                    branch[i] = this_branch + [i]
                    queue.append(i)
                    layers[i] = layer
                elif any(j in branch[i] for j in this_branch if j != root):
                    continue
                else:
                    this_cycle = this_branch + branch[i][::-1]
                    if all((c_i, c_j) in G.edges for c_i, c_j in zip(this_cycle[:-1], this_cycle[1:])):
                        this_cycle.pop()
                        this_cycle = tuple(sort_cycle(this_cycle))
                        
                        if any_sub_cycle(bonds, this_cycle):
                            continue
                        if need_check:
                            if not check_func(this_cycle):
                                continue
                            
                        
                        if this_cycle not in total_cycles:
                            for c_i, c_j in zip(this_cycle[:-1], this_cycle[1:]):
                                G.edges[c_i, c_j]['n'] += 1
                                if G.edges[c_i, c_j]['n'] == 2:
                                    G.remove_edge(c_i, c_j)
                            G.edges[this_cycle[-1], this_cycle[0]]['n'] += 1
                            if G.edges[this_cycle[-1], this_cycle[0]]['n'] == 2:
                                G.remove_edge(this_cycle[-1], this_cycle[0])
                            total_cycles.add(this_cycle)
                            added += 1
                        
        for l in range(max_ring_elements//2):
            if added >= len(G[root]):
                break
            for n in range(len(queue)):
                if len(G[root]):
                    find_next_layer(queue.pop(0), l)

    return total_cycles

def constrain_cycles(positions: np.ndarray, cycles: set[tuple[int, ...]]) -> tuple[np.ndarray, set[tuple[int, ...]]]:
    counts = np.zeros(positions.shape[0], dtype=int)

    for cycle in cycles:
        counts[list(cycle)] += 1
    
    valid = np.nonzero(counts >= 3)[0]
    valid_cycles = set(filter(lambda x: all(i in valid for i in x), cycles))
    valid_positions = positions[valid]

    return valid_positions, valid_cycles

def visualize_cycles(name: str,
                     positions: np.ndarray, 
                     cycles: set[tuple[int]], 
                     boundaries: list[tuple[int]] | None = None, 
                     isolated_nodes: set[int] | None = None, 
                     cell: np.ndarray | None = None, 
                     pbcs: list[bool] = [False, False, False],
                     show_labels: bool = False,
                     ):
    cycles_elements = np.array(list(map(lambda x: len(x), cycles)))
    centers = np.array(list(map(lambda x: middle_point(positions, x, cell, pbcs), cycles)))
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if cell is None:
        bound = positions.max(axis=0)[:2]
        uv = np.array([[0, 0], [bound[0], 0], [bound[0], bound[1]], [0, bound[1]], [0, 0]])
    elif cell.ndim == 1:
        bound = cell[:2]
        uv = np.array([[0, 0], [bound[0], 0], [bound[0], bound[1]], [0, bound[1]], [0, 0]])
    elif cell.ndim == 2:
        bound = cell.copy()
        
        uv = np.array([[0, 0], 
                       [bound[0, 0], 0], 
                       [bound[0, 0] + bound[1, 0], bound[1, 1]], 
                       [bound[1, 0], bound[1, 1]], 
                       [0, 0]])

    ax.set_xlim(uv[:, 0].min() - 1, uv[:, 0].max() + 1)
    ax.set_ylim(uv[:, 1].min() - 1, uv[:, 1].max() + 1)
    
    ax.plot(uv[:, 0], uv[:, 1], c='black')
    
    # Plot atoms
    scatter = ax.scatter(positions[:, 0], positions[:, 1], s=5, c='black')
    
    
    
    if show_labels:
        cursor = mplcursors.cursor(scatter, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            idx = sel.target.index
            sel.annotation.set_text(f"{idx}")
    
    # Plot cycles
    sc1 = ax.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], s=10, c=cycles_elements, cmap='rainbow', vmin = 3, vmax = 8)
    ax.set_title(name)
    
    if show_labels:
        cursor2 = mplcursors.cursor(sc1, hover=True)
        @cursor2.connect("add")
        def on_add(sel):
            idx = sel.target.index
            sel.annotation.set_text(f"[{idx}]")

    # Plot boundaries
    if boundaries is not None:
        for boundary in boundaries:
            bound_pos = positions[list(boundary)]
            delta = np.roll(bound_pos, 1, axis=0) - bound_pos
            delta = map_pbc_vector(delta, cell, pbcs)
            for i in range(len(boundary)):
                ax.arrow(bound_pos[i, 0], 
                        bound_pos[i, 1], 
                        delta[i, 0], 
                        delta[i, 1], 
                        head_width=0.2, head_length=0.2, fc='gray', ec='gray')
    
    # Plot extra nodes
    if isolated_nodes is not None and len(isolated_nodes) > 0:
        sc2 = ax.scatter(positions[list(isolated_nodes), 0], positions[list(isolated_nodes), 1], s=10, c='maroon', label='marked')

    handles1, labels1 = sc1.legend_elements()
    handles2 = [plt.Line2D([0], [0], marker='o', color='w', label='marked', markerfacecolor='maroon', markersize=8)]
    labels2 = ['marked']
    ax.legend(handles1 + handles2, labels1 + labels2, loc='lower left')
    plt.show()
    plt.close()

def cycle_area(positions: np.ndarray, cycle: tuple, cell: np.ndarray | None, pbcs: list[bool] = [False, False, False], append_last: bool = True):
    if append_last:
        pos = positions[list(cycle) + [cycle[0]]]
    else:
        pos = positions[list(cycle)]
    
    delta = pos - pos[0]
    delta = map_pbc_vector(delta, cell, pbcs)
    
    pos = pos[0] + delta
    
    area = 0.5 * np.abs(np.dot(pos[:,0], np.roll(pos[:,1], 1)) - np.dot(pos[:,1], np.roll(pos[:,0], 1))) 
    
    return area

def middle_point(positions: np.ndarray, cycle: tuple, cell: np.ndarray | None, pbcs: list[bool] = [False, False, False]):
    positions = positions[list(cycle)]
    if cell is None or not any(pbcs):
        return positions.mean(axis=0)
    elif cell.ndim == 1:
        cell = np.diag(cell)
    elif cell.ndim == 2:
        pass 
    else:
        raise ValueError("cell should be a (3,) or (3, 3)")
    
    inv_cell = np.linalg.inv(cell)
    inv_pos = positions @ inv_cell
    
    z_ave = np.empty(3, dtype=complex)
    x_ave = np.empty(3)
    for i in range(3):
        if pbcs[i]:
            z_ave[i] = np.exp(1j * 2 * np.pi * inv_pos[:, i]).mean()
            x_ave[i] = np.angle(z_ave[i]) / (2 * np.pi)
        else:
            x_ave[i] = inv_pos[:, i].mean()
        
    x_ave = np.dot(x_ave % 1, cell)
    return x_ave