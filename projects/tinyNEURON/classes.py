
class Compartment:
    def __init__(self):
        pass

class Morphology:
    def __init__(self):
        pass

class SVCLoader:
    def __init__(self, swc_path):
        self.swc_path=swc_path

    def parse_into_morphology(self):
        f = open(self.swc_path, 'r')
        lines=f.readlines()
        valid_rows=[x.strip().split() for x in lines if x[0]!='#']
        print(valid_rows[:100])


if __name__=='__main__':
    loader=SVCLoader('/home/maria/MITNeuralComputation/projects/tinyNEURON/6-Som-3d-trace-B.CNG.swc')
    morphology=loader.parse_into_morphology()