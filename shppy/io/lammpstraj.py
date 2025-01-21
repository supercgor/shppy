import numpy as np
import re

from pathlib import Path
from multiprocessing import Pool
from io import StringIO
from functools import partial

from ..atom import Atoms
from ..utils import get_workers
        
__key_dtype_dict = {"id": int, "type": int, "x": float, "y": float, "z": float, "ix": float, "iy": float, "iz": float, "vx": float, "vy": float, "vz": float, "q": float, "fx": float, "fy": float, "fz": float, "c_q[1]": float, "c_q[2]": float, "c_q[3]": float, "c_q[4]": float}
        
def read_timestep(path: str | Path) -> np.ndarray:
    with open(path, "r") as f:
        l = f.read()
        t = re.findall(r"ITEM: TIMESTEP\s+(\d+)", l)
    return np.array(t, dtype = int)

def read_lammps_dump_text(path, index = 0, sort_by_id = True, type_map = None, num_workers = 1):
    num_workers = get_workers(num_workers)
    
    if isinstance(index, int) and index >= 0:
        with open(path, "r") as f:
            for _ in range(index):
                seek_pos = _skip_one_frame(f)
                f.seek(seek_pos)
            return _read_one_frame(f, sort_by_id = sort_by_id, type_map = type_map)
        
    elif isinstance(index, slice) and index.stop > 0 and num_workers == 1:
        step = index.step or 1
        start = index.start or 0
        stop = index.stop
        frames = []
        with open(path, "r") as f:
            for i in range(start, stop, step):
                for _ in range(step - 1):
                    seek_pos = _skip_one_frame(f)
                    f.seek(seek_pos)
                frames.append(_read_one_frame(f, sort_by_id = sort_by_id, type_map = type_map))
        return frames
    
    else:
        with open(path, "r") as f:
            lines = f.read()
            offsets = [m.start() for m in re.finditer(r"ITEM: TIMESTEP", lines)]
        
        offsets = offsets[index]
                        
        if isinstance(offsets, list):
            lines = [StringIO(lines[i:j]) for i, j in zip(offsets[:-1], offsets[1:])] + [StringIO(lines[offsets[-1]:])]
            fn = partial(_read_one_frame, sort_by_id = sort_by_id, type_map = type_map)
            if num_workers > 1:
                with Pool(num_workers) as p:
                    frames = p.map(fn, lines)
            else:
                frames = [fn(l) for l in lines]
                
            return frames
        
        else:
            return _read_one_frame(StringIO(lines[offsets:]), sort_by_id = sort_by_id, type_map = type_map)
        
def _skip_one_frame(lines):
    this_frame = True
    seek_pos = 0
    while True:
        l = lines.readline()
        if l.startswith("ITEM: TIMESTEP"):
            if this_frame:
                this_frame = False
                seek_pos += len(l)
            else:
                return seek_pos
        else:
            seek_pos += len(l)
            
def _read_one_frame(lines, sort_by_id = False, type_map = None):    
    reader = iter(lines)
        
    while True:                
        l = next(reader, None)
                
        if l.startswith("ITEM: TIMESTEP"):
            t = int(next(reader))
        
        elif l.startswith("ITEM: NUMBER OF ATOMS"):
            n_atoms = int(next(reader))
        
        elif l.startswith("ITEM: BOX BOUNDS"):
            bounds = l.strip()[17:].split()
            box_params = list(filter(lambda x: "xy" in x or "xz" in x or "yz" in x, bounds))
            box_bounds = list(filter(lambda x: "p" in x or "f" in x or "s" in x, bounds))
            cell = np.zeros((3, 3))
            pbcs = []
            if len(box_params) == 0:
                xlo, xhi = map(float, next(reader).strip().split())
                ylo, yhi = map(float, next(reader).strip().split())
                zlo, zhi = map(float, next(reader).strip().split())
                celldisp = np.array([xlo, ylo, zlo])
                cell = np.diag([xhi - xlo, yhi - ylo, zhi - zlo])
                
            elif len(box_params) == 3:
                xlo, xhi, xy = map(float, next(reader).strip().split())
                ylo, yhi, xz = map(float, next(reader).strip().split())
                zlo, zhi, yz = map(float, next(reader).strip().split())
                celldisp = np.array([xlo, ylo, zlo])
                cell = np.array([[xhi - xlo, 0, 0], 
                                 [xy, yhi - ylo, 0], 
                                 [xz, yz, zhi - zlo]])
            
            else:
                raise ValueError(f"Invalid box params: {box_params}")
            
            if len(box_bounds) == 3:
                for i in range(3):
                    if "p" in box_bounds[i]:
                        pbcs.append(True)
                    else:
                        pbcs.append(False)
            
            else:
                raise ValueError(f"Invalid box bounds: {box_bounds}")
        
        elif l.startswith("ITEM: ATOMS"):
            keys = l.strip()[12:].split()
            convert_dict = {i: __key_dtype_dict[keys[i]] for i in range(len(keys))}
            use_col = [i for i in range(len(keys)) if keys[i] not in ["ix", "iy", "iz"]]
            
            data = np.loadtxt(reader, converters=convert_dict, usecols=use_col, max_rows=n_atoms)
            
            if "id" in keys:
                ids = data[:, keys.index("id")]
                if sort_by_id:
                    sort_order = np.argsort(ids)
                    ids = ids[sort_order]
                    data = data[sort_order]
            else:
                ids = np.arange(n_atoms)
            
            if "type" in keys:
                types = data[:, keys.index("type")]
                if type_map is not None:
                    types = np.vectorize(type_map.get)(types)
            else:
                types = np.zeros(n_atoms, dtype=int)

            if "x" in keys:
                positions = data[:, (keys.index("x"), keys.index("y"), keys.index("z"))]
                scaled_positions = None
            elif "xu" in keys:
                positions = data[:, (keys.index("xu"), keys.index("yu"), keys.index("zu"))]
                scaled_positions = None
            elif "xs" in keys:
                positions = None
                scaled_positions = data[:, (keys.index("xs"), keys.index("ys"), keys.index("zs"))]
            elif "xsu" in keys:
                positions = None
                scaled_positions = data[:, (keys.index("xsu"), keys.index("ysu"), keys.index("zsu"))]
            else:
                raise ValueError(f"Invalid keys: {keys}")
            
            if "vx" in keys:
                velocities = data[:, (keys.index("vx"), keys.index("vy"), keys.index("vz"))]
            else:
                velocities = None
            if "q" in keys:
                charges = data[:, keys.index("q")]
            else:
                charges = None
            if "fx" in keys:
                forces = data[:, (keys.index("fx"), keys.index("fy"), keys.index("fz"))]
            else:
                forces = None
            if "c_q[1]" in keys:
                quaternions = data[:, (keys.index("c_q[1]"), keys.index("c_q[2]"), keys.index("c_q[3]"), keys.index("c_q[4]"))]
            else:
                quaternions = None
            
            atoms = Atoms(numbers=types,
                          positions=positions, 
                          cell=cell, 
                          pbc=pbcs, 
                          celldisp=celldisp, 
                          velocities=velocities, 
                          charges=charges,
                          scaled_positions=scaled_positions,
                        )
            atoms.info["timestep"] = t
            atoms.set_array("id", ids)

            if forces is not None:
                atoms.set_array("forces", forces)
            if quaternions is not None:
                atoms.set_array("quaternions", quaternions)

            return atoms
        
        else:
            raise ValueError(f"Invalid line: {l}")