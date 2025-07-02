from abc import ABC, abstractmethod

class EigenSolverInterface(ABC):
    @abstractmethod
    def get_effective_indices(self) -> list: pass

    @abstractmethod
    def get_mode_field(self, mode_index=0): pass

    @abstractmethod
    def update_parameter(self, key: str, value): pass

    @abstractmethod
    def solve(self): pass

    @abstractmethod
    def export_results(self, path: str): pass

    @abstractmethod
    def summary(self) -> str: pass

