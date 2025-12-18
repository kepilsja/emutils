from abc import ABC, abstractmethod

class Simulation(ABC):
    @abstractmethod
    def set_geometry(self, geometry):
        raise NotImplementedError()

    @abstractmethod
    def set_materials(self, materials):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @abstractmethod
    def export_results(self, path: str):
        raise NotImplementedError()

    @abstractmethod
    def summary(self):
        raise NotImplementedError()

class EigenSolverInterface(Simulation):
    def _is_component_valid(self, component):
        if not isinstance(component, str):
            raise TypeError(f'Invalid typ of "component", expected string, got {type(component)}')
        valid_components = ('ex', 'ey', 'ez', 'e2',
                            'hx', 'hy', 'hz', 'h2')
        return True if component in valid_components else False

    @abstractmethod
    def get_effective_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_mode_field(self, mode_index=0, component='e2'):
        raise NotImplementedError()

    @abstractmethod
    def get_mode_farfield(self, mode_index=0):
        raise NotImplementedError()

    @abstractmethod
    def export_mode_farfield(self, filename, mode_index=0):
        raise NotImplementedError()

class WGModel:
    def __init__(self, model: Simulation):
        self.model = model
