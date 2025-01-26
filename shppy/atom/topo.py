import networkx as nx
import numpy as np
from collections import defaultdict, Counter

from typing import Literal
from numpy.typing import NDArray    

from .basic import radius_query_kdtree

def find_radius_cycles(rs: NDArray[np.float_], cutoff: float = 3.6, max_length: int = 8, mode: Literal["rule", "none"] = "rule", pbc: NDArray[np.bool_] | bool = False, cell: NDArray[np.float_] | None = None):
    nei = radius_query_kdtree(rs, k = 5, cutoff = cutoff, pbc = pbc, cell = cell)[1][:,1:]
    
    G = nx.Graph()
    
    G.add_nodes_from(range(len(rs)))
    
    edges = [(i, int(j), {'count': 0}) for i, neighbors in enumerate(nei) for j in neighbors if i < j and j != len(rs)]

    G.add_edges_from(edges)

    cycles = sorted(nx.chordless_cycles(G, max_length), key=lambda x: len(x)) # type: ignore
    
    if mode == "none":
        
        return cycles, G
    
    elif mode == "rule":
        outs = []
        ats = defaultdict(list)
        for i, cycle in enumerate(cycles):
            counter = Counter()
            for atom in cycle:
                counter.update(ats[atom])
            
            if len(counter) > 0:    
                max_shared = 0
                for k, v in counter.items():
                    if v == 3 and len(cycles[k]) < len(cycle):
                        if max_shared == 3:
                            max_shared = 4
                            break
                        else:
                            max_shared = 3
                    elif v > 3:
                        max_shared = v
                        break
                        
            else:
                max_shared = 0
            
            if max_shared <= 3:
                outs.append(cycle)
                for atom in cycle:
                    ats[atom].append(i)
            
        return outs, G
