from .base import EigenSolverInterface

class LumericalModel(EigenSolverInterface):
    def set_geometry(self, geometry):
        raise NotImplementedError()

    def set_materials(self, materials):
        raise NotImplementedError()

    def run(self):
        self

    def export_results(self, path: str):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError