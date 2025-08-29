
from collections import defaultdict

class Compartment:
    def __init__(self, neighbors):
        self.neighbors=neighbors

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

    def parse_into_morphology(self):
        f = open(self.swc_path, 'r')
        lines=f.readlines()
        valid_rows=[x.strip().split() for x in lines if x[0]!='#']
        print(valid_rows[:100])
        n_d=self.build_neighbors_dct(valid_rows)
        compartments=[Compartment(n_d[k]) for k in n_d.keys()]
        return compartments


if __name__=='__main__':
    loader=SVCLoader('/home/maria/MITNeuralComputation/projects/tinyNEURON/6-Som-3d-trace-B.CNG.swc')
    morphology=loader.parse_into_morphology()
    print(morphology)