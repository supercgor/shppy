import shppy
import numpy as np

print(shppy.add_floats(1.0, 2.0))

from shppy.atom import Atoms
# from shppy.io import _read_one_frame

# atoms = Atoms(numbers = np.array([1,2,3]), positions = np.array([[1,2,3], [4,5,6], [7,8,9]]))

f = open("/Users/supercgor/Library/CloudStorage/OneDrive-個人/Documents/Data/simu/disordered-pbc/t120-traj1.pos", "r")
out_f = []

for _ in range(10):
    out_f.append(next(f))

print(out_f)


# atoms = _read_one_frame(f, sort_by_id = True, type_map = None)
# print(atoms)