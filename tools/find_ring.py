import argparse
import numpy as np
import pandas as pd

from functools import partial
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count

from shppy.atom.topo import radius_query_kdtree, find_radius_cycles
from shppy.atom import Atoms, pbc_map
from shppy import io

def parse_args():
    parser = argparse.ArgumentParser(description="Find circle")
    parser.add_argument("input", type=str, help="Path to the input file.")
    parser.add_argument("-m", "--mode", type=str, help="Mode", default="h6", choices=["h6", "tau4", "rings"])
    parser.add_argument("-o", "--output", type=str, help="path to the output file.", default="out/cycles-count.csv")
    
    parser.add_argument("--mask-island", action="store_true", help="Mask island.")
    parser.add_argument("--save-atoms", action="store_true", help="Save atoms.")
    
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite if the output file exists.")
    parser.add_argument("-j", "--workers", type=int, help="Number of workers.", default=-1)
    return parser.parse_args()

def process_h6(args, atoms):
    cycles, _ = find_radius_cycles(atoms.positions, 
                                   cutoff = 3.6, 
                                   max_length = 8, 
                                   mode = "rule", 
                                   pbc = atoms.get_pbc(), 
                                   cell = atoms.get_cell().array)
    
    counts = defaultdict(list)

    for cycle in cycles:
        for c in cycle:
            counts[c].append(len(cycle))
    
    h6 = np.zeros(len(atoms))

    for k, v in counts.items():
        v = np.array(v)
        h6[k] = 1 / 3 * np.sum( 1 / np.power(2, np.abs(v - 6)))
        
    atoms.set_array("h6", h6)
    if args.mask_island:
        cellx, celly = atoms.get_cell().array.diagonal()[:2]
        mask = (atoms.positions[:, 0] - cellx / 2) ** 2 + (atoms.positions[:, 1] - celly / 2) ** 2 < ((cellx + celly) / 12) ** 2
        atoms.set_array("mask", mask)
    
    return atoms

def process_rings(args, atoms):
    cycles, _ = find_radius_cycles(atoms.positions, 
                                   cutoff = 3.6, 
                                   max_length = 8, 
                                   mode = "rule", 
                                   pbc = atoms.get_pbc(), 
                                   cell = atoms.get_cell().array)
    
    MR = defaultdict(int)

    for cycle in cycles:
        MR[len(cycle)] += 1
    
    return MR

def process_tau4(args, atoms: Atoms):
    pos = atoms.positions[atoms.numbers == 8]
    nei = radius_query_kdtree(pos, cutoff = 3.6, k = 5, cell = atoms.cell.array, pbc = atoms.pbc)[1][:,1:]
    mask = nei < len(pos)
    bond_num = mask.sum(axis=1)
    
    pos = np.concatenate([pos, [[0.0, 0.0, 0.0]]], axis = 0)
    nei_pos = pos[nei] # N x k x 3
    
    ij = nei_pos - pos[:-1, None]
    
    # 3-bond delta
    bond3delta = ij[bond_num == 3][:, :3].sum(axis=1) # N x 3
    bond3delta = bond3delta / np.linalg.norm(bond3delta, axis=-1, keepdims=True)
    ij[bond_num == 3][:, 3] = bond3delta
    
    ij = pbc_map(ij, atoms.cell.array, atoms.pbc, align_center=True)
    ij = ij / np.linalg.norm(ij, axis = -1, keepdims=True)
    
    ij = ij[:,None] * ij[:,:,None] # N x k x k x 3
    
    u1, u2 = np.triu_indices(n=4, k=1)
        
    mask = np.logical_and(mask[:, None], mask[:, :, None])
    
    mask = mask[:, u1, u2]
    
    ij = ij[: , u1, u2] # N x k * (k-1) /2 x 3
    ij = ij.sum(-1)
    ij = np.clip(ij, a_min = -1, a_max = 1)
    ij = np.arccos(ij)
    ij[~mask] = - np.inf

    ij_top2 = np.argpartition(ij, -2, axis=1)[:, -2:]  # 每行的倒數兩個索引
    ij_top2 = np.take_along_axis(ij, ij_top2, axis=1)
    
    # ij_top2 = np.where(ij_top2 < np.arccos(-1/3), np.pi, ij_top2)
    ij_top2 = np.clip(ij_top2, a_min = np.arccos(-1/3), a_max = np.pi)
    # ij_top2 = np.clip(ij_top2, a_min = 0, a_max = np.pi)
    
    alpha, beta = ij_top2[:, 0], ij_top2[:, 1]
        
    tau4 = (2 * np.pi - (alpha + beta)) / (2 * np.pi - 2 * np.arccos(-1/3))
    
    atoms.set_array("tau4", tau4)
    
    if args.mask_island:
        cellx, celly = atoms.get_cell().array.diagonal()[:2]
        mask = (atoms.positions[:, 0] - cellx / 2) ** 2 + (atoms.positions[:, 1] - celly / 2) ** 2 < ((cellx + celly) / 12) ** 2
        atoms.set_array("mask", mask)

    return atoms
    

def main(args):
    in_pth = Path(args.input)
    out_path = Path(args.output)
    name = in_pth.with_suffix("").name
    
    if out_path.is_dir() or out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = (out_path / "count.csv")
        
    elif out_path.is_file() or out_path.suffix != "":
        if out_path.exists():
            if not args.force:
                if input(f"{out_path} exists, overwrite? (y/n)").lower() not in ["y", "yes", "t", "true", "1"]:
                    print("Abort!")
                    exit()
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        
    print(f"[INFO] Reading {in_pth}")
    
    atoms_lst = io.read(in_pth, format="extxyz", index=slice(-1))
    
    if args.workers == -1:
        args.workers = cpu_count()
        
    print(f"[INFO] Processing in {args.mode} mode, {args.workers} workers, {len(atoms_lst)} frames")
    
    if args.mode == "rings":
        fn = partial(process_rings, args)
        if args.workers == 1:
            results = list(map(fn, atoms_lst))
        else:
            with Pool(args.workers) as pool:
                results = pool.map(fn, atoms_lst)
        
        keys = list(range(3, 11))
        
        if out_path.exists():
            df = pd.read_csv(out_path, index_col=0)
            for i, result in enumerate(results):
                df.loc[f"{name}-ts{i}"] = [result[k] for k in keys]
        else:
            data = {k: [result[k] for result in results] for k in keys}
            df = pd.DataFrame(data, index=[f"{name}-ts{i}" for i in range(len(results))])
    
    elif args.mode == "tau4":
        fn = partial(process_tau4, args)
        if args.workers == 1:
            results = list(map(fn, atoms_lst))
        else:
            with Pool(args.workers) as pool:
                results = pool.map(fn, atoms_lst)
        
        keys = ["tau4_mean", "tau4_std"]

        if out_path.exists():
            df = pd.read_csv(out_path, index_col = 0)
        else:
            df = pd.DataFrame(index=[f"{name}-ts{i}" for i in range(len(results))], columns=keys)
            
        for i, result in enumerate(results):
            tau4 = result.get_array("tau4")
            if args.mask_island:
                mask = result.get_array("mask")
                tau4_mean = tau4[mask].mean()
                tau4_std = tau4[mask].std()
            else:
                tau4_mean = tau4.mean()
                tau4_std = tau4.std()
            df.loc[f"{name}-ts{i}", "tau4_mean"] = tau4_mean
            df.loc[f"{name}-ts{i}", "tau4_std"] = tau4_std

        if args.save_atoms:
            io.write(out_path.with_name(f"{name}_tau4").with_suffix(".xyz"), results, format="extxyz")
            
    elif args.mode == "h6":
        fn = partial(process_h6, args)
        if args.workers == 1:
            results = list(map(fn, atoms_lst))
        else:
            with Pool(args.workers) as pool:
                results = pool.map(fn, atoms_lst)
        
        keys = ["h6_mean", "h6_std"]
            
        if out_path.exists():
            df = pd.read_csv(out_path, index_col=0)
        else:
            df = pd.DataFrame(index=[f"{name}-ts{i}" for i in range(len(results))], columns=keys)
            
        for i, result in enumerate(results):
            h6 = result.get_array("h6")
            if args.mask_island:
                mask = result.get_array("mask")
                h6_mean = h6[mask].mean()
                h6_std = h6[mask].std()
            else:
                h6_mean = h6.mean()
                h6_std = h6.std()
            
            df.loc[f"{name}-ts{i}", "h6_mean"] = h6_mean
            df.loc[f"{name}-ts{i}", "h6_std"] = h6_std

        if args.save_atoms:
            io.write(out_path.with_name(f"{name}_h6").with_suffix(".xyz"), results, format="extxyz")
            
    df.to_csv(out_path)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)