import lumapi
import numpy as np

from .. import um, nm
from ..materials import si_palik, ge_palik
from ..util import dotdict

DEFAULTS = dotdict(
    substrate_width = 30*um,
    substrate_height = 20*um,
    core_width = 2*um,
    core_height = 1*um,
    etch_depth = 500*nm,
    materials = dotdict(
        core = ge_palik,
        substrate = si_palik,
    ),
    solver_x_span = 15*um,
    solver_y_span = 10*um,
)
DEFAULTS.materials.film = DEFAULTS.materials.core

class BasicWaveguide(lumapi.MODE):
    def __init__(self, filename=None, key=None, hide=False, serverArgs=None,
                 remoteArgs=None):
        serverArgs = serverArgs or dict()
        remoteArgs = remoteArgs or dict()
        super().__init__(filename, key, hide, serverArgs, remoteArgs)

        self.geometry = dotdict()

    def _add_base_ridge_geometry(self, core_width, core_height, materials):
        
        self.geometry.substrate = self.addrect(
            name='substrate',
            x=0, x_span=DEFAULTS.substrate_width,
            y_max=0, y_min=-DEFAULTS.substrate_height,
            material=materials['substrate']
        )
        self.geometry.core = self.addrect(
            name='core',
            x=0, x_span=core_width,
            y_max=core_height, y_min=0,
            material=DEFAULTS.materials.core
        )
    
    def _add_solver(self):
        self.solver = self.addfde(
            x=0, x_span = DEFAULTS.solver_x_span,
            y=0, y_span = DEFAULTS.solver_y_span,
            x_min_bc = 'PML',
            x_max_bc = 'PML',
            y_min_bc = 'PML',
            y_max_bc = 'PML',
            mesh_cells_x = 100,
            mesh_cells_y = 100
        )

    def _add_core_mesh(self):
        self.core_mesh = self.addmesh(
            set_mesh_multiplier = 1,
            x_mesh_multiplier = 7,
            y_mesh_multiplier = 7,
            based_on_a_structure = 1,
            structure = 'core',
            buffer = 1*um
        )

class Ridge(BasicWaveguide):
    def __init__(self, core_width=DEFAULTS.core_width,core_height=DEFAULTS.core_height,
                 materials=DEFAULTS.materials, **kwargs):
        super().__init__(**kwargs)

        self._add_base_ridge_geometry(core_width, core_height, materials)
        self._add_solver()
        self._add_core_mesh()

class Rib(BasicWaveguide):
    def __init__(self, core_width=DEFAULTS.core_width, core_height=DEFAULTS.core_height,
                 etch_depth=DEFAULTS.etch_depth, materials=DEFAULTS.materials, **kwargs):
        super().__init__(**kwargs)

        self._add_base_ridge_geometry(core_width, core_height, materials)
        self.geometry.film = self.addrect(
            name='film',
            x=0, x_span=DEFAULTS.substrate_width,
            y_max=self.geometry.core.y_max-etch_depth, y_min=0,
            material=materials['film']
        )

        self._add_solver()
        self._add_core_mesh()
