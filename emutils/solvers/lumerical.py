from .base import EigenSolverInterface
from ..lumapi_loader import add_lumapi_to_path

try:
    add_lumapi_to_path()

    import lumapi # type: ignore

    class LumericalModel(lumapi.MODE, EigenSolverInterface):
        def set_geometry(self, geometry):
            raise NotImplementedError()

        def set_materials(self, materials):
            raise NotImplementedError()

        def run(self):
            super().findmodes()

        def export_results(self, path: str):
            raise NotImplementedError

        def summary(self):
            raise NotImplementedError

        def get_effective_indices(self):
            raise NotImplementedError

        def get_mode_field(self, mode_index=0): 
            raise NotImplementedError
        
        def get_mode_farfield(self, mode_index=1, resolution=(300,300),
                            ambient_index=1.0): 
            if mode_index < 1:
                raise ValueError(f'Modes are indexed starting from 1, got {mode_index}')
            monitor = f'FDE::data::mode{mode_index}'
            na, nb = resolution

            dataset = self.getresult(monitor, "E")
            dataset["H"] = self.getattribute(self.getresult(monitor,"H"),"H")
            dataset["Lumerical_dataset"]['attributes'] = ["E", "H"]

            Esqr = self.farfield3d(dataset, 1, na, nb, ambient_index)
            ux = self.farfieldux(dataset, 1, na, nb, ambient_index)
            uy = self.farfielduy(dataset, 1, na, nb, ambient_index)
        
            return ux, uy, Esqr

        def export_mode_farfield(self, filename, **kwargs):
            ux, uy, farfield = self.get_mode_farfield(**kwargs)
            self.h5write(filename, "ux", ux, "overwrite")
            self.h5write(filename, "uy", uy)
            self.h5write(filename, "farfield_E2", farfield)
except ImportError:
    pass
