import os
import logging
import tempfile
import atexit
from dataclasses import dataclass, field, fields
from typing import Union, Tuple, Dict, Iterable, Callable, Optional, List
import numbers
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from ..lumer.lumapi_loader import add_lumapi_to_path

logger = logging.getLogger(__name__)

# mapping have to be swapped since this is varFDTD
POLARIZATION_MAP = {'tm': 'E mode (TE)', 'te': 'H mode (TM)'}
um = 1e-6
nm = 1e-9

try:
    add_lumapi_to_path()
    import lumapi
    from lumapi import MODE as LumerMODE

except ImportError as err:
    logger.warning("Lumerical unavailable!")
    logger.warning(err)

    LumerMODE = object


def _is_air(material):
    return material.lower() == 'air'


@dataclass
class GratingGeometry:
    """
    Represents the geometry of a photonic grating coupler.

    Attributes:
        layers_stack (Dict[str, Tuple[str, float]]):
            A dictionary mapping layer names to a tuple of (material_name, thickness in meters).
            Layers should be ordered from bottom to top! Must include 'core',
            'substrate', and 'cladding' layers. Layers made of 'air' are not drawn
            (background material is used instead).

        grating_shape (Iterable[float]):
            A sequence of floats describing consecutive sections of a single grating
            segment. Its meaning depends on `core_only`:
            - core_only=True: remaining core height of each section measured from
              the core bottom, e.g. (150e-9, 220e-9) for 70 nm shallow etch of 220 nm core.
            - core_only=False: etch depth of each section measured down from the top
              of the `etch_from` layer (0 means not etched). The etch may go through
              any number of layers, e.g. through the top cladding and partially
              into the core, or deep into the substrate.

        pitch (float):
            The period of the grating (in meters).

        duty_cycle (Union[float, Iterable[float]]):
            Ratio(s) defining the proportions of the segments lengths in coupler section.
            If an iterable, must sum to 1 and match the number of sections
            in a single segment.

        n_segments (int):
            Number of segments in the grating. Must be greater than 1.

        input_wg_length (float):
            Length of the input waveguide before the grating. Must be positive.

        core_only (bool):
            If True (default) the grating is defined only in the core layer and the
            trenches are filled with the material of the layer directly above the core.

        etch_from (str, optional):
            Only used if core_only=False. Name of the layer from whose top surface the
            etch starts. Defaults to the topmost non-air layer.

        fill_material (str, optional):
            Only used if core_only=False. Material filling etched trenches. Defaults to
            the material of the layer directly above `etch_from` ('air' if none).

    Notes:
        - Every attribute change is validated (and reverted if invalid) and then
          `_callback(key, value)` is called (if `_callback` is set).
        - Use `update(**kwargs)` to change several dependent attributes at once
          (e.g. `grating_shape` together with `duty_cycle`).
    """
    layers_stack: Dict[str, Tuple[str, float]]
    grating_shape: Iterable[float]
    pitch: float
    duty_cycle: Union[float, Iterable[float]]
    n_segments: int
    input_wg_length: float
    core_only: bool = True
    etch_from: Optional[str] = None
    fill_material: Optional[str] = None

    _callback: Callable[[str, object], None] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, '_ready', True)

    def __setattr__(self, key, value):
        if key.startswith('_') or not self.__dict__.get('_ready'):
            super().__setattr__(key, value)
            return
        self.update(**{key: value})

    def update(self, **kwargs):
        """Set several attributes at once, validate them together and trigger a single callback."""
        field_names = {f.name for f in fields(self) if not f.name.startswith('_')}
        unknown = kwargs.keys() - field_names
        if unknown:
            raise AttributeError(f"Unknown geometry parameters: {unknown}")

        old = {key: getattr(self, key) for key in kwargs}
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        try:
            self._validate()
        except Exception:
            for key, value in old.items():
                object.__setattr__(self, key, value)
            raise

        if self._callback:
            if len(kwargs) == 1:
                self._callback(*next(iter(kwargs.items())))
            else:
                self._callback('update', kwargs)

    def _validate(self):
        required_layers = {'core', 'substrate', 'cladding'}
        missing_layers = required_layers - self.layers_stack.keys()
        if missing_layers:
            raise ValueError(f"layer_stack is missing required layers: {missing_layers}")

        for layer, (material, thickness) in self.layers_stack.items():
            if not isinstance(material, str):
                raise TypeError(f"Layer '{layer}' material name must be a string.")
            if not isinstance(thickness, numbers.Number) or thickness <= 0:
                raise ValueError(f"Layer '{layer}' thickness must be a positive number.")

        if not isinstance(self.n_segments, int) or self.n_segments <= 1:
            raise ValueError("n_segments must be an integer greater than 1.")

        if not isinstance(self.input_wg_length, numbers.Number) or self.input_wg_length <= 0:
            raise ValueError("input_wg_length must be a positive number.")

        if not isinstance(self.pitch, numbers.Number) or self.pitch <= 0:
            raise ValueError("pitch must be a positive number.")

        self._validate_duty_cycle()
        if self.core_only:
            self._validate_core_only_shape()
        else:
            self._validate_etch_depths()

    def _validate_duty_cycle(self):
        if isinstance(self.duty_cycle, numbers.Number):
            if not (0.0 <= self.duty_cycle <= 1.0):
                raise ValueError(f"duty_cycle value has to be number between 0 and 1, got {self.duty_cycle}")
            object.__setattr__(self, 'duty_cycle', [self.duty_cycle, 1.0 - self.duty_cycle])
        elif isinstance(self.duty_cycle, Iterable):
            object.__setattr__(self, 'duty_cycle', list(self.duty_cycle))
            total = sum(self.duty_cycle)
            if not abs(total - 1.0) < 1e-6:
                raise ValueError(f"duty_cycle values must sum to 1, got {total}.")
        else:
            raise TypeError("duty_cycle must be a float or an iterable of floats.")

        n_sections = len(list(self.grating_shape))
        if len(self.duty_cycle) != n_sections:
            raise ValueError(
                f"duty_cycle must match length of grating_shape. "
                f"Got {len(self.duty_cycle)} and {n_sections}."
            )

    def _validate_core_only_shape(self):
        core_thickness = self.layers_stack['core'][1]
        for i, height in enumerate(self.grating_shape):
            if not (0 <= height <= core_thickness):
                raise ValueError(
                    f"Grating height at grating_shape[{i}] = {height} must be non-negative "
                    f"and less than core thickness ({core_thickness})."
                )

    def _validate_etch_depths(self):
        if self.etch_from is not None and self.etch_from not in self.layers_stack:
            raise ValueError(f"etch_from layer '{self.etch_from}' is not defined in layers_stack.")
        if self.fill_material is not None and not isinstance(self.fill_material, str):
            raise TypeError("fill_material must be a string.")

        etch_from = self._etch_from_layer()
        max_depth = self.layer_bounds()[etch_from][1]
        for i, depth in enumerate(self.grating_shape):
            if not (0 <= depth <= max_depth):
                raise ValueError(
                    f"Etch depth at grating_shape[{i}] = {depth} must be non-negative "
                    f"and not exceed the stack thickness below top of '{etch_from}' ({max_depth})."
                )

    def _etch_from_layer(self):
        if self.core_only:
            return 'core'
        if self.etch_from is not None:
            return self.etch_from
        non_air = [name for name, (material, _) in self.layers_stack.items() if not _is_air(material)]
        return non_air[-1]

    def layer_bounds(self, buffer=0.0):
        """Returns {layer_name: (y_min, y_max)}, bottom and top layers are extended by `buffer`."""
        bounds = {}
        y = 0.0
        last = len(self.layers_stack) - 1
        for i, (name, (_, thickness)) in enumerate(self.layers_stack.items()):
            thickness += buffer * ((i == 0) + (i == last))
            bounds[name] = (y, y + thickness)
            y += thickness
        return bounds

    def section_surfaces(self, buffer=0.0):
        """
        Returns (y_top, fill_material, surfaces) where y_top is the level from which
        the etch starts, fill_material fills the trenches and surfaces[j] is the level
        of the etched surface of j-th grating section.
        """
        etch_from = self._etch_from_layer()
        names = list(self.layers_stack)
        idx = names.index(etch_from)
        y_top = self.layer_bounds(buffer)[etch_from][1]

        if self.core_only:
            fill = None
            depths = [self.layers_stack['core'][1] - height for height in self.grating_shape]
        else:
            fill = self.fill_material
            depths = list(self.grating_shape)
        if fill is None:
            fill = self.layers_stack[names[idx + 1]][0] if idx + 1 < len(names) else 'air'

        return y_top, fill, [y_top - depth for depth in depths]


@dataclass(frozen=True)
class Rect:
    name: str
    material: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    in_grating: bool = False


def build_rects(geom: GratingGeometry, buffer: float) -> List[Rect]:
    """
    Builds the list of non-overlapping rectangles representing the whole grating coupler
    cross-section. Grating starts at x=0, input waveguide spans x < 0.
    """
    y_top, fill, surfaces = geom.section_surfaces(buffer)

    # x-intervals as (label, x_min, x_max, surface level)
    intervals = [('input', -buffer - geom.input_wg_length, 0.0, y_top)]
    x = 0.0
    for i in range(geom.n_segments):
        for j, (ff, surface) in enumerate(zip(geom.duty_cycle, surfaces)):
            if ff == 0:
                continue
            intervals.append((f'seg{i}_sec{j}', x, x + ff * geom.pitch, surface))
            x += ff * geom.pitch
    intervals.append(('tail', x, x + buffer, y_top))

    rects = []
    for name, (y_min, y_max) in geom.layer_bounds(buffer).items():
        material = geom.layers_stack[name][0]
        if _is_air(material):
            continue
        above_etch = y_min >= y_top
        # merge consecutive intervals with equal clipped layer top
        runs = []
        for label, x0, x1, surface in intervals:
            top = y_max if above_etch else min(y_max, surface)
            if runs and np.isclose(runs[-1][3], top, rtol=0, atol=1e-15):
                runs[-1][2] = x1
                runs[-1][4] = label
            else:
                runs.append([label, x0, x1, top, label])
        if len(runs) == 1:
            _, x0, x1, top, _ = runs[0]
            rects.append(Rect(name, material, x0, x1, y_min, top))
            continue
        for first, x0, x1, top, last in runs:
            if top <= y_min:
                continue
            label = first if first == last else f'{first}-{last}'
            rects.append(Rect(f'{name}_{label}', material, x0, x1, y_min, top, in_grating=True))

    if not _is_air(fill):
        for label, x0, x1, surface in intervals:
            if surface < y_top:
                rects.append(Rect(f'fill_{label}', fill, x0, x1, surface, y_top, in_grating=True))

    return rects


class GratingCoupler(LumerMODE):
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
        duty_cycle (Union[float, Iterable[float]], optional):
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
            waveguide. Defaults to 'out'.
        theta (float, optional):
            The angle in degrees for the Gaussian source, required for the 'in'
            configuration. Defaults to 20.
        source_wl_range (Tuple[float, float], optional):
            The start and end wavelengths for the simulation source in meters.
            Defaults to (1450*nm, 1650*nm).
        filename (Union[str, Path], optional):
            Path to save the Lumerical project file. If None, a temporary
            file is created and managed automatically. Defaults to None.
        **kwargs:
            Additional keyword arguments passed to the parent `LumerMODE` class.

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
    def __init__(self,
            layers_stack=dict(
                substrate=('Si (Silicon) - Palik', 0.5*um),
                box=('SiO2 (Glass) - Palik', 2.0*um),
                core=('Si (Silicon) - Palik', 220*nm),
                oxide_cladding=('SiO2 (Glass) - Palik', 2.0*um),
                cladding=('air', 2.0*um)
            ),
            grating_shape=(150*nm, 220*nm),
            pitch=660*nm,
            duty_cycle=0.45,
            n_segments=25,
            input_wg_length=8*um,
            core_only=True,
            etch_from=None,
            fill_material=None,
            polarization='te',
            configuration='in',
            theta=20,
            source_wl_range=(1450*nm, 1650*nm),
            filename:Union[str, Path]=None,
            **kwargs
        ):
        if LumerMODE is object:
            err_msg = "Module 'lumapi' is required to use this functionality!"
            logger.error(err_msg)
            raise ImportError(err_msg)
        
        super().__init__(**kwargs)
        
        self._Z_SPAN = 50*um
        self._BUFFER = 2*um
        self._FREQ_POINTS = 81
        self._BEAM_WAIST_RADIUS = 10*um
        self._IS_INITIALIZED = False

        self.polarization = polarization
        self.theta = theta
        self.configuration = configuration
        self.source_wl_range = source_wl_range

        # Instantiate ModelParameters with callback
        self.geom = GratingGeometry(
            layers_stack, grating_shape, pitch, duty_cycle, n_segments, input_wg_length,
            core_only=core_only, etch_from=etch_from, fill_material=fill_material,
            _callback=self._on_param_change,
        )

        self._setup_save_location(filename)
        self._initialize_objects()

    def _on_param_change(self, name, value):
        # logger.info(f"Parameter '{name}' changed to {value}. Reinitializing geometry...")
        self._IS_INITIALIZED = False
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
    
    @property
    def polarization(self):
        return self._polarization
    
    @polarization.setter
    def polarization(self, value):
        if value.lower() not in ['te', 'tm']:
            raise ValueError(f'Invalid polarization, "TE" or "TM" available, got "{value}"')
        self._polarization = value
        if self._IS_INITIALIZED:
            self.solver.polarization = POLARIZATION_MAP[value.lower()]

    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, value):
        if not (value is None or isinstance(value, numbers.Number)):
            raise TypeError(f'Parameter theta should be None or numeric value, got {type(value)}')
        self._theta = value
        if self._IS_INITIALIZED:
            self.source.angle_theta = -self.theta
    
    @property
    def configuration(self):
        return self._configuration
    
    @configuration.setter
    def configuration(self, value):
        if value not in ['in', 'out']:
            raise ValueError(f'Invalid configiration, "in" or "out" available, got "{value}"')
        if value=='in' and not isinstance(self._theta, numbers.Number):
            raise ValueError('Provide theta angle while in input configuration of the coupler')
        self._configuration = value

        if self._IS_INITIALIZED:
            self.select('source')
            self.delete()
            self._add_source()
    
    @property
    def source_wl_range(self):
        return self._source_wl_range
    
    @source_wl_range.setter
    def source_wl_range(self, wl_range):
        if len(wl_range)!=2:
            raise ValueError('Invalid source wavelength range')
        self._source_wl_range = sorted(wl_range)
        
        if self._IS_INITIALIZED:
            self.source.wavelength_start = self.source_wl_range[0]
            self.source.wavelength_stop = self.source_wl_range[1]

            self.solver.simulation_wavelength_min = self.source_wl_range[0]
            self.solver.simulation_wavelength_max = self.source_wl_range[1]

    def _initialize_objects(self):
        self.switchtolayout()
        self.deleteall()

        self._create_geometry()
        self._add_fdtd_solver()
        self._add_source()
        self._add_monitor()

        self._IS_INITIALIZED = True

    def _create_geometry(self):
        self.addstructuregroup(name='grating')
        for rect in build_rects(self.geom, self._BUFFER):
            self.addrect(
                name=rect.name,
                material=rect.material,
                x=(rect.x_min + rect.x_max) / 2,
                x_span=rect.x_max - rect.x_min,
                y=(rect.y_min + rect.y_max) / 2,
                y_span=rect.y_max - rect.y_min,
                z_span=self._Z_SPAN
            )
            if rect.in_grating:
                self.addtogroup('grating')

    def _add_fdtd_solver(self):
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
            polarization=POLARIZATION_MAP[self.polarization],
            mesh_accuracy=5
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
        
    def _setup_save_location(self, filename):
        self._tempdir = None
        if filename:
            path = Path(filename).with_suffix('.lms')
            self._savepath = path.parent
            self._filename = Path(path.name)
            self._savepath.mkdir(parents=True, exist_ok=True)
        else:
            # Create a temporary directory for simulation
            self._tempdir = tempfile.TemporaryDirectory()
            atexit.register(self._cleanup_tempdir)  # Clean up after run
            self._savepath = Path(self._tempdir.name)
            self._filename = Path("temp_grating_model.lms")
        super().save(str(self._savepath / self._filename))

    def _cleanup_tempdir(self):
        if self._tempdir:
            self._tempdir.cleanup()

    def get_results(self):
        """
        Retrieve transmission results from the 'flux monitor' DFT monitor and compute relevant performance metrics.

        This method extracts the transmission spectrum and computes:
        - Wavelengths and corresponding transmission (absolute value)
        - Frequencies (calculated from wavelengths)
        - Peak transmission value and its corresponding wavelength and frequency
        - 1 dB and 3 dB bandwidths, both in the wavelength domain and optionally in the spectral (frequency) domain
        - Fit parameters of a Gaussian fit to the transmission spectrum (if fitting is successful)

        Returns:
            dict: A dictionary containing the following keys:
                - 'wavelength': np.ndarray of wavelengths (in meters)
                - 'frequency': np.ndarray of corresponding frequencies (in Hz)
                - 'transmission': np.ndarray of absolute transmission values
                - 'peak': dict with keys:
                    - 'max_transmission': float, maximum transmission value
                    - 'wavelength': float, wavelength (in meters) of the maximum transmission
                    - 'frequency': float, frequency (in Hz) of the maximum transmission
                - 'bandwidth': dict with keys:
                    - '1dB': float or None, 1 dB bandwidth (in meters or GHz based on `spectral`)
                    - '3dB': float or None, 3 dB bandwidth (in meters or GHz based on `spectral`)
                - 'fit_params': list or None, optimized fit parameters (if fitting succeeded)

        Raises:
            RuntimeError: If the result data is unavailable or improperly structured.
        """
        try:
            results = self.getresult('flux monitor', 'T')
        except Exception as e:
            raise RuntimeError(f"Could not fetch result from monitor: {e}")

        if results is None or 'lambda' not in results or 'T' not in results:
            raise RuntimeError("Result does not contain expected data.")

        wavelength = np.squeeze(results['lambda'])  # in meters
        frequency = np.squeeze(results['f'])
        transmission = np.abs(np.squeeze(results['T']))

        # Peak transmission info
        max_idx = np.argmax(transmission)
        max_transmission = transmission[max_idx]
        peak_wavelength = wavelength[max_idx]
        peak_frequency = frequency[max_idx]

        # --- Fit Gaussian or Lorentzian to the transmission spectrum ---

        def gaussian(x, a, x0, sigma, offset):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset
        
        def find_bandwidth(dB_drop):
            target = max_transmission * 10**(-dB_drop / 10)
            wavelengths = np.linspace(xdata.min(), xdata.max(), 3000)
            transmission = gaussian(wavelengths, *popt)
            indices = np.where(transmission >= target)[0]
            if len(indices) >= 2:
                low = wavelengths[indices[0]]/nm
                high = wavelengths[indices[-1]]/nm
                return high - low
            return None

        # Normalize for fitting
        xdata = wavelength
        ydata = transmission
        try:
            popt, _ = curve_fit(gaussian, xdata, ydata,
                                p0=[max_transmission, peak_wavelength, 0.1e-6, min(ydata)])
            bandwidth_1dB = find_bandwidth(1)
            bandwidth_3dB = find_bandwidth(3)
        except RuntimeError:
            popt = None
            bandwidth_1dB = None
            bandwidth_3dB = None

        return {
            'wavelength': wavelength,
            'frequency': frequency,
            'transmission': transmission,
            'peak': {
                'max_transmission': max_transmission,
                'wavelength': peak_wavelength,
                'frequency': peak_frequency
            },
            'bandwidth': {
                '1dB': bandwidth_1dB,
                '3dB': bandwidth_3dB
            },
            'fit_params': popt
        }
    def save_results(self, filename="results.npz"):
        """
        Save simulation results and current model parameters to a .npz file.
        
        Args:
            filename (str): Path to the .npz file to save.
        """
        results = self.get_results()
        
        # Flatten the results dictionary
        flat = {}
        for key, val in results.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    flat[f"{key}_{subkey}"] = subval
            else:
                flat[key] = val

        # Add current model parameters from the dataclass instance self.params
        for key, val in self.geom.__dict__.items():
            if not key.startswith('_'):
                flat[key] = val

        np.savez(filename, **flat)