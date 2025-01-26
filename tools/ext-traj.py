#!/usr/bin/env python3

import argparse
import numpy as np

from pathlib import Path
from ast import literal_eval

from shppy.io import read, read_timestep, write

def get_parser():
    parser = argparse.ArgumentParser(description="Convert mW to TIP4P")
    parser.add_argument("input", type=str, help="Path to the input file.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file.", default="out")
    
    parser.add_argument("--show-ts", action="store_true", help="Show timestep range.")
    
    parser.add_argument("--show-z", action="store_true", help="Show z range.")
    parser.add_argument("--bin", type=int, help="Number of bins to show", default=100)
    
    parser.add_argument("--index", type=str, help="Index to extract, e.g. 0 or 0:10:2", default=":")
    parser.add_argument("--format", type=str, help="Output format", default="extxyz", choices=["extxyz"])
    
    parser.add_argument("--ts-range", type=str, help="Timestep range to extract, e.g. [-inf, inf]", default="")
    parser.add_argument("--z-range", type=str, help="z range to extract, e.g. [-inf, inf]", default="")
    parser.add_argument("--filter-type", type=str, help="Filter type", default="[8, ]")

    parser.add_argument("-f", "--force", help="Force overwrite the output file.", action="store_true")
    
    return parser

def read_num(s):
    if s == "inf":
        return np.inf
    elif s == "-inf":
        return -np.inf
    else:
        return float(s)

def read_min_max(s):
    if "[" in s:
        s = s.lstrip("[").rstrip("]")
        return map(read_num, s.split(","))
    elif s:
        return read_num(s), np.inf
    else:
        return -np.inf, np.inf

def main(args):
    in_pth = Path(args.input)
    
    if ":" in args.index:
        if args.index == ":":
            index = slice(-1)
        else:
            index = slice(*map(int, args.index.split(":")))
    else:
        index = int(args.index)
    
    if args.show_ts:
        ts = read_timestep(in_pth)[index]
        print(f"""[INFO] Timestep range: {ts.min()} - {ts.max()}
+--------+-----------+
| index  | timestep  |
+========+===========+
{chr(10).join(f"| {i:>6} | {t:>9} |" for i, t in enumerate(ts))}
+--------+-----------+""")
        exit()
        
    if args.show_z:
        if isinstance(index, slice):
            atoms = read(in_pth, 0)
        else:
            atoms = read(in_pth, index)
        zs = atoms.positions[:, 2]
        zs, edges = np.histogram(zs, bins=args.bin, density=True)
        print(f"""[INFO] z range: {edges[0]} - {edges[-1]}
+---------+-----------+
|  start  |     z     |
+=========+===========+
{chr(10).join(f"| {c:>7.2f} | {z:>9.3%} |" for c, z in zip(edges[:-1], zs))}
+---------+-----------+""")
        exit()
        
    name = in_pth.with_suffix("").name
        
    out_pth = Path(args.output)
        
    if out_pth.is_dir() or out_pth.suffix == "":
        out_pth.mkdir(parents=True, exist_ok=True)
        out_pth = out_pth / name
        
    elif out_pth.is_file() or out_pth.suffix != "":
        if out_pth.exists():
            if not args.force:
                if input(f"{out_pth} exists, overwrite? (y/n)").lower() not in ["y", "yes", "t", "true", "1"]:
                    print("Abort!")
                    exit()
        else:
            out_pth.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading {in_pth}")
    
    atoms_lst = read(in_pth, index, type_map={1: 8, 2: 1})
    
    if not isinstance(atoms_lst, list):
        atoms_lst = [atoms_lst]
    
    print(f"[INFO] Processing {len(atoms_lst)} frames")
    
    ts_min, ts_max = read_min_max(args.ts_range)
    z_min, z_max = read_min_max(args.z_range)
    filter_type = literal_eval(args.filter_type)
    
    if ts_min != -np.inf:
        atoms_lst = map(lambda atoms: atoms[atoms.info['timestep'] >= ts_min], atoms_lst)
        
    if ts_max != np.inf:
        atoms_lst = map(lambda atoms: atoms[atoms.info['timestep'] <= ts_max], atoms_lst)
    
    if filter_type:
        atoms_lst = map(lambda atoms: atoms[np.isin(atoms.numbers, filter_type)], atoms_lst)
    
    if z_min != -np.inf:
        atoms_lst = map(lambda atoms: atoms[atoms.positions[:, 2] > z_min], atoms_lst)
        
    if z_max != np.inf:
        atoms_lst = map(lambda atoms: atoms[atoms.positions[:, 2] < z_max], atoms_lst)
    
    atoms_lst = list(atoms_lst)
        
    if args.format == "extxyz":
        write(out_pth.with_suffix(".xyz"), atoms_lst, format=args.format)
    else:
        raise ValueError(f"Unsupported format: {args.format}")
        
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
