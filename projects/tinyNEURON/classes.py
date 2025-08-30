from collections import defaultdict
import numpy as np

class Compartment:
    def __init__(self, ix, xyz, r, type_):
        self.ix = ix
        self.xyz = xyz
        self.r = r
        self.type = type_
        self.neighbors = []  # list of Compartment objects
        self.R_ax = {}       # dict: neighbor_ix → axial resistance

    def __repr__(self):
        return f"Compartment(ix={self.ix}, xyz={self.xyz}, r={self.r:.2f})"

class Morphology:
    def __init__(self, compartments):
        self.compartments = compartments

class SVCLoader:
    def __init__(self, swc_path):
        self.swc_path = swc_path

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
        return {
            int(row[0]): (float(row[2]), float(row[3]), float(row[4]))
            for row in swc_rows
        }

    def get_radius(self, swc_rows):
        return {
            int(row[0]): float(row[5])  # note: radius is column 5 (not 4)
            for row in swc_rows
        }

    def get_compartment_type(self, swc_rows):
        return {
            int(row[0]): int(row[1])
            for row in swc_rows
        }

    def compute_axial_resistance(self, xyz1, r1, xyz2, r2, Ra=100):
        """
        Compute axial resistance between two compartments using average radius.
        Ra: specific axial resistance [ohm*cm]
        """
        L = np.linalg.norm(np.array(xyz1) - np.array(xyz2))
        r_avg = (r1 + r2) / 2
        if r_avg == 0:
            return np.inf  # avoid divide-by-zero
        return (Ra * L) / (np.pi * r_avg**2)

    def parse_into_morphology(self):
        with open(self.swc_path, 'r') as f:
            lines = f.readlines()

        valid_rows = [x.strip().split() for x in lines if x[0] != '#']
        neighbors_dct = self.build_neighbors_dct(valid_rows)
        xyz_dct = self.get_coordinates(valid_rows)
        r_dct = self.get_radius(valid_rows)
        type_dct = self.get_compartment_type(valid_rows)

        # Create compartments
        compartments = {
            ix: Compartment(ix, xyz_dct[ix], r_dct[ix], type_dct[ix])
            for ix in neighbors_dct.keys()
        }

        # Link neighbors and compute pairwise resistances
        for ix, comp in compartments.items():
            for n_ix in neighbors_dct[ix]:
                neighbor = compartments[n_ix]
                comp.neighbors.append(neighbor)

                # Compute and store axial resistance
                R_ax = self.compute_axial_resistance(
                    comp.xyz, comp.r, neighbor.xyz, neighbor.r
                )
                comp.R_ax[n_ix] = R_ax  # resistance to neighbor
                # Optional: symmetry check / duplicate not needed if both sides store

        return Morphology(list(compartments.values()))

# === Entry point ===
if __name__ == '__main__':
    loader = SVCLoader('/home/maria/MITNeuralComputation/projects/tinyNEURON/6-Som-3d-trace-B.CNG.swc')
    morphology = loader.parse_into_morphology()
    
    # Show first 10 compartments with resistance dicts
    for comp in morphology.compartments[:10]:
        print(f"{comp}")
        for n_ix, R in comp.R_ax.items():
            print(f"  → neighbor {n_ix} with R_ax = {R:.3f}")
