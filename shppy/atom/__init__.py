from ase import Atoms as ASEAtoms

class Atoms(ASEAtoms):    
    @property
    def xs(self):
        return self.positions[:, 0]
    
    @property
    def ys(self):
        return self.positions[:, 1]
    
    @property
    def zs(self):
        return self.positions[:, 2]
