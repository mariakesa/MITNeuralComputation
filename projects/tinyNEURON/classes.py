
from collections import defaultdict
import numpy as np

class Compartment:
    def __init__(self, ix, xyz):
        self.neighbors=[]
        self.ix=ix
        self.xyz=xyz
        self.L=0

    def compute_length(self):
        L = 0
        for n in self.neighbors:
            diff = np.array(self.xyz) - np.array(n.xyz)
            L += np.linalg.norm(diff)
        return L

    def __repr__(self):
        return f"Compartment(ix={self.ix}, xyz={self.xyz}, L={self.L:.2f})"

class Morphology:
    def __init__(self):
        pass

class SVCLoader:
    def __init__(self, swc_path):
        self.swc_path=swc_path

    def build_neighbors_dct(self, swc_rows):
        neighbors = defaultdict(list)
        for row in swc_rows:
            node_id = int(row[0])
            parent_id = int(row[-1])
            if parent_id != -1:
                neighbors[node_id].append(parent_id)
                neighbors[parent_id].append(node_id)
        return dict(neighbors)
    
    def get_coordinates(self, swc_rows):
        xyz_dct={}
        for row in swc_rows:
            node_id = int(row[0])
            x=float(row[2])
            y=float(row[3])
            z=float(row[4])
            xyz_dct[node_id]=(x,y,z)
        return xyz_dct

    def parse_into_morphology(self):
        f = open(self.swc_path, 'r')
        lines=f.readlines()
        valid_rows=[x.strip().split() for x in lines if x[0]!='#']
        n_d=self.build_neighbors_dct(valid_rows)
        xyz_dct=self.get_coordinates(valid_rows)
        print(xyz_dct)
        print(n_d)
        naive_comp=[]
        for k in n_d.keys():
            comp=Compartment(k, xyz_dct[k])
            naive_comp.append(comp)
        ix_to_comp = {c.ix: c for c in naive_comp}
        for comp in naive_comp:
            comp.neighbors = [ix_to_comp[n_ix] for n_ix in n_d[comp.ix]]
        for comp in naive_comp:
            comp.L=comp.compute_length()
        compartments=naive_comp
        return compartments


if __name__=='__main__':
    loader=SVCLoader('/home/maria/MITNeuralComputation/projects/tinyNEURON/6-Som-3d-trace-B.CNG.swc')
    morphology=loader.parse_into_morphology()
    print(morphology)