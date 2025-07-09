from abc import ABC, abstractmethod

class Simulation(ABC):
    @abstractmethod
    def set_geometry(self, geometry): pass

    @abstractmethod
    def set_materials(self, materials): pass

    @abstractmethod
    def run(self): pass

    @abstractmethod
    def export_results(self, path: str): pass

    @abstractmethod
    def summary(self): pass

class EigenSolverInterface(Simulation):
    @abstractmethod
    def get_effective_indices(self): pass

    @abstractmethod
    def get_mode_field(self, mode_index=0): pass

    @abstractmethod
    def export_mode_farfield(self, filename, mode_index=0): pass

class WGModel:
    def __init__(self, model: Simulation):
        self.model = model
