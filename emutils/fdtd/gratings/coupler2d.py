"""2D grating coupler simulated with varFDTD in Lumerical MODE."""
from typing import Union
from pathlib import Path

import numpy as np

from ... import um, nm
from ._lumapi import LumerMODE
from .base import _GratingCouplerBase
from .cross_section import build_rects
from .geometry import GratingGeometry, DEFAULT_LAYERS_STACK

# mapping have to be swapped since this is varFDTD
POLARIZATION_MAP = {'tm': 'E mode (TE)', 'te': 'H mode (TM)'}


class GratingCoupler2D(_GratingCouplerBase, LumerMODE):
    """Automates the creation and simulation of a photonic grating coupler in Lumerical MODE.

    This class provides a high-level interface to build the geometry, configure
    the FDTD solver, add sources and monitors, run the simulation, and process
    the results for a 2D grating coupler. It is designed to be reactive,
    automatically reinitializing the simulation environment when key geometric
    or simulation parameters are changed.

    The default parameters implement a standard SOI (Silicon-on-Insulator) grating
    coupler geometry as described in "Silicon Photonics Design: From Devices to
    Systems" by Chrostowski L. and Hochberg M., chapter 5.2.

    Args:
        layers_stack (Dict[str, Tuple[str, float]], optional):
            Defines the material layers from bottom to top. Each entry is a
            tuple of (material_name, thickness_in_meters). Defaults to a
            standard SOI stack. Must include layers named `substrate`, `core`
            and `cladding`.
        grating_shape (Tuple[float, ...], optional):
            A sequence of floats defining consecutive sections of a single grating
            period. With `core_only=True` these are remaining core heights, otherwise
            etch depths measured from the top of `etch_from` layer.
            Defaults to (150*nm, 220*nm).
        pitch (float, optional):
            The period of the grating in meters. Defaults to 660*nm.
        fill_factor (Union[float, Iterable[float]], optional):
            The ratio(s) of the segment lengths within a single pitch. If a float,
            it assumes a two-segment grating. If an iterable, its length must
            match `grating_shape` and its values must sum to 1. Defaults to 0.45.
        n_segments (int, optional):
            The total number of grating periods. Defaults to 25.
        input_wg_length (float, optional):
            The length of the input waveguide before the grating region in meters.
            Defaults to 8*um.
        core_only (bool, optional):
            If True, the grating is defined only in the core layer. If False, the
            grating is etched from the top of `etch_from` layer through any number
            of layers below it. Defaults to True.
        etch_from (str, optional):
            Name of the layer the etch starts from (only with `core_only=False`).
            Defaults to the topmost non-air layer.
        fill_material (str, optional):
            Material filling the etched trenches (only with `core_only=False`).
            Defaults to the material of the layer directly above `etch_from`.
        polarization (str, optional):
            The polarization of the light, either 'te' or 'tm'. Defaults to 'te'.
        configuration (str, optional):
            The simulation setup, either 'in' for coupling light into the
            waveguide from free space, or 'out' for coupling light out of the
            waveguide. Defaults to 'in'.
        theta (float, optional):
            The angle in degrees for the Gaussian source, required for the 'in'
            configuration. Defaults to 20.
        source_wl_range (Tuple[float, float], optional):
            The start and end wavelengths for the simulation source in meters.
            Defaults to (1450*nm, 1650*nm).
        save_as (Union[str, Path], optional):
            Path to save the Lumerical project file. If None, a temporary
            file is created and managed automatically. Defaults to None.
        **kwargs:
            Additional keyword arguments passed to the parent `LumerMODE` class
            (e.g. `filename` of the project to open, `hide`).

    Attributes:
        geom (GratingGeometry): An object holding the validated geometric parameters.
            Changing attributes on this object will trigger a simulation re-build.
        solver (object): The Lumerical FDTD solver object.
        source (object): The Lumerical source object (ModeSource or GaussianSource).
        polarization (str): Light polarization ('te' or 'tm').
        theta (float): Source angle for 'in' configuration.
        configuration (str): Simulation type ('in' or 'out').
        source_wl_range (Tuple[float, float]): The wavelength range of the simulation.
        sx (float): The total simulation width in the x-direction.
        sy (float): The total simulation height in the y-direction.

    Raises:
        ImportError: If the 'lumapi' dependency required for Lumerical
            interoperability is not installed.
        ValueError: If invalid values are provided for parameters like
            `polarization` or `configuration`.
    """
    _FILE_SUFFIX = '.lms'
    _LUMERICAL_CLASS = LumerMODE

    def __init__(self,
            layers_stack=DEFAULT_LAYERS_STACK,
            grating_shape=(150*nm, 220*nm),
            pitch=660*nm,
            fill_factor=0.45,
            n_segments=25,
            input_wg_length=8*um,
            core_only=True,
            etch_from=None,
            fill_material=None,
            polarization='te',
            configuration='in',
            theta=20,
            source_wl_range=(1450*nm, 1650*nm),
            save_as:Union[str, Path]=None,
            **kwargs
        ):
        self._check_lumapi()
        super().__init__(**kwargs)

        self._Z_SPAN = 50*um
        self._BUFFER = 2*um
        self._FREQ_POINTS = 81
        self._BEAM_WAIST_RADIUS = 10*um

        self._init_parameters(polarization, theta, configuration, source_wl_range)

        # Instantiate ModelParameters with callback
        self.geom = GratingGeometry(
            dict(layers_stack), grating_shape, pitch, fill_factor, n_segments, input_wg_length,
            core_only=core_only, etch_from=etch_from, fill_material=fill_material,
            _callback=self._on_param_change,
        )

        self._setup_save_location(save_as)
        self._initialize_objects()

    @property
    def sx(self):
        return self.geom.input_wg_length + self.geom.n_segments * self.geom.pitch + 2*self._BUFFER

    @property
    def sy(self):
        return sum(height for name, (mat, height) in self.geom.layers_stack.items()) + 2*self._BUFFER

    @property
    def x_sim_center(self):
        return self.sx/2 - self.geom.input_wg_length - self._BUFFER

    @property
    def y_sim_center(self):
        return self.sy/2

    @property
    def y_core(self):
        y_min, y_max = self.geom.layer_bounds(self._BUFFER)['core']
        return (y_min + y_max) / 2

    @property
    def source_loc(self):
        return self._source_loc

    @source_loc.setter
    def source_loc(self, loc):
        x, y = loc
        self._source_loc = (x, y)
        if self._IS_INITIALIZED:
            self.source.x = x
            self.source.y = y

    @property
    def monitor_loc(self):
        return self._monitor_loc

    @monitor_loc.setter
    def monitor_loc(self, loc):
        x, y = loc
        self._monitor_loc = (x, y)
        if self._IS_INITIALIZED:
            self.select('flux monitor')
            self.set('x', x)
            self.set('y', y)

    def _apply_polarization(self):
        self.solver.polarization = POLARIZATION_MAP[self.polarization.lower()]

    def _apply_wl_range(self):
        self.source.wavelength_start = self.source_wl_range[0]
        self.source.wavelength_stop = self.source_wl_range[1]

        self.solver.simulation_wavelength_min = self.source_wl_range[0]
        self.solver.simulation_wavelength_max = self.source_wl_range[1]

    def _create_geometry(self):
        self.addstructuregroup(name='grating')
        for rect in build_rects(self.geom, self._BUFFER):
            # material passed as addrect argument is not applied by lumapi
            obj = self.addrect(
                name=rect.name,
                x=(rect.x_min + rect.x_max) / 2,
                x_span=rect.x_max - rect.x_min,
                y=(rect.y_min + rect.y_max) / 2,
                y_span=rect.y_max - rect.y_min,
                z_span=self._Z_SPAN
            )
            obj.material = rect.material
            if rect.in_grating:
                self.addtogroup('grating')

    def _add_solver(self):
        self.solver = self.addvarfdtd(
            x=self.x_sim_center,
            x_span=self.sx - 2 * self._BUFFER,
            y=self.y_sim_center,
            y_span=self.sy - 2 * self._BUFFER,
            z=0,
            z_span=self._Z_SPAN - 1*um,
            x0=-self.geom.input_wg_length / 2,
            y0=self.y_core - self.y_sim_center,
            number_of_test_points=1,
            test_points=np.array([[
                -self.geom.input_wg_length / 2,
                0.8 * self.geom.layers_stack['core'][1] + self.y_core - self.y_sim_center
            ]]),
            polarization=POLARIZATION_MAP[self.polarization.lower()],
            mesh_accuracy=5,
            simulation_time=15000e-15,
        )
        self.solver.set_simulation_bandwidth = 1
        self.solver.simulation_wavelength_min = self.source_wl_range[0]
        self.solver.simulation_wavelength_max = self.source_wl_range[1]

    def _add_source(self):
        if self.configuration=='in':
            self.source_loc = (
                5*um, self.sy - 1.1*self._BUFFER
            )
            self.source = self.addgaussian(
                injection_axis='y',
                direction='backward',
                x=self.source_loc[0],
                x_span=self.sx,
                y=self.source_loc[1],
                angle_theta=-self._theta,
                waist_radius_w0=self._BEAM_WAIST_RADIUS,
                set_wavelength=True,
                wavelength_start=self.source_wl_range[0],
                wavelength_stop=self.source_wl_range[1],
            )
        elif self.configuration=='out':
            self.source_loc = (
                -0.9*self.geom.input_wg_length,
                self.y_core
            )
            self.source = self.addmodesource(
                x=self.source_loc[0],
                y=self.source_loc[1],
                y_span=7*self.geom.layers_stack['core'][1],
                wavelength_start=self.source_wl_range[0],
                wavelength_stop=self.source_wl_range[1],
            )

    def _add_monitor(self):
        self.adddftmonitor()

        self.set('name', 'flux monitor')
        self.set('override global monitor settings', 1)
        self.set('frequency points', self._FREQ_POINTS)

        if self.configuration=='in':
            self.set('monitor type', 5)  # 2D X-normal
            self.set('x', -0.9 * self.geom.input_wg_length)
            self.set('y', self.y_core)
            self.set('y span', 7 * self.geom.layers_stack['core'][1])

        if self.configuration=='out':
            self.set('monitor type', 6)  # 2D Y-normal
            self.set('y', self.sy - self._BUFFER - 0.1*self.geom.layers_stack['cladding'][1])
            self.set('x', self.x_sim_center)
            self.set('x span', self.sx - 2*self._BUFFER)

    def _fetch_transmission(self):
        return self._monitor_transmission('flux monitor')
