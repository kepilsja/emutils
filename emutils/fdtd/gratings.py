import os
import logging
import tempfile
import atexit
from dataclasses import dataclass, field
from typing import Union, Tuple, Dict, Iterable, Callable
import numbers
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from ..lumapi_loader import setup_lumapi

logger = logging.getLogger(__name__)

# mapping have to be swapped since this is varFDTD
POLARIZATION_MAP = {'tm': 'E mode (TE)', 'te': 'H mode (TM)'}
um = 1e-6
nm = 1e-9

try:
    setup_lumapi()
    import lumapi
    from lumapi import MODE as LumerMODE

except ImportError as err:
    logger.warning("Lumerical unavailable!")
    logger.warning(err)

    LumerMODE = object


@dataclass
class GratingGeometry:
    """
    Represents the geometry of a photonic grating coupler.

    Attributes:
        layer_stack (Dict[str, Tuple[str, float]]): 
            A dictionary mapping layer names to a tuple of (material_name, thickness in microns).
            Layers should be ordered from bottom to top! Must include 'core',
            'substrate', and 'cladding' layers.

        grating_shape (Iterable[float]): 
            A sequence of floats describing the grating segment consecutive core heights,
            e.g. (1e-6, 0.5e-6) translates to grating section built from
            segments with heights of 1 and 0.5 microns with lengths specified by
            values of `pitch` and `duty_cycle` parameters

        pitch (float): 
            The period of the grating (in microns).

        duty_cycle (Union[float, Iterable[float]]): 
            Ratio(s) defining the proportions of the segments lengths in coupler section.
            If an iterable, must sum to 1 and match the number of sections 
            in a single segment.

        n_segments (int): 
            Number of segments in the grating. Must be greater than 1.

        input_wg_length (float): 
            Length of the input waveguide before the grating. Must be positive.

    Notes:
        - Automatically calls `_callback(key, value)` on attribute change (if `_callback` is set).
    """
    layers_stack: Dict[str, Tuple[str, float]]
    grating_shape: Iterable[float]
    pitch: float
    duty_cycle: Union[float, Iterable[float]]
    n_segments: int
    input_wg_length: float

    _callback: Callable[[str, object], None] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._validate()

    def __setattr__(self, key, value): # TODO: How to validate on parameter change?
        super().__setattr__(key, value)
        if key != "_callback" and hasattr(self, "_callback") and self._callback:
            self._callback(key, value)

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

        grating_shape_list = list(self.grating_shape)
        for i, height in enumerate(grating_shape_list):
            if not (0 <= height <= self.layers_stack['core'][1]):
                raise ValueError(
                    f"Grating height at grating_shape[{i}] = {height} must be non-negative "
                    f"and less than core thickness ({self.layers_stack['core'][1]})."
                )

        if isinstance(self.duty_cycle, Iterable):
            duty_list = list(self.duty_cycle)
            if len(duty_list) != len(grating_shape_list):
                raise ValueError(
                    f"duty_cycle must match length of grating_shape. "
                    f"Got {len(duty_list)} and {len(grating_shape_list)}."
                )
            total = sum(duty_list)
            if not abs(total - 1.0) < 1e-6:
                raise ValueError(f"duty_cycle values must sum to 1, got {total}.")
        elif isinstance(self.duty_cycle, numbers.Number):
            if not (0.0 <= self.duty_cycle <= 1.0):
                raise ValueError(f"duty_cycle value has to be number between 0 and 1, got {self.duty_cycle}")
            super().__setattr__('duty_cycle', [self.duty_cycle, 1.0 - self.duty_cycle])
            
        else:
            raise TypeError("duty_cycle must be a float or an iterable of floats.")


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
            A sequence of floats defining the consecutive heights of the core
            layer within a single grating period. Defaults to (150*nm, 220*nm).
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
        geom (ModelGeometry): An object holding the validated geometric parameters.
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
        return self._y_core
    
    @y_core.setter
    def y_core(self, value):
        self._y_core = value

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
        x_span = self.sx
        y_spans = np.array([height for (_, height) in self.geom.layers_stack.values()])
        y_spans[0] += self._BUFFER
        y_spans[-1] += self._BUFFER
        ys = np.cumsum(y_spans) - y_spans/2
        adjust_next_layer_thickess = False
        for layer, y, thickness in zip(self.geom.layers_stack, ys, y_spans):
            if layer=='core':
                self.y_core = y
                adjust_next_layer_thickess = True
                continue
            if adjust_next_layer_thickess:
                thickness += self.geom.layers_stack['core'][1]
                y -= self.geom.layers_stack['core'][1]/2
                adjust_next_layer_thickess = False
            if self.geom.layers_stack[layer][0]=='air':
                continue
            self.addrect(
                name=layer,
                material=self.geom.layers_stack[layer][0],
                x=self.x_sim_center,
                x_span=x_span,
                y=y,
                y_span=thickness,
                z_span=self._Z_SPAN
            )
        self._add_grating()

    def _add_grating(self):
        if self.layoutmode() == 0:
            self.switchtolayout()
        
        # input waveguide
        self.addrect(
                name='core',
                material=self.geom.layers_stack['core'][0],
                x_min=-self._BUFFER - self.geom.input_wg_length,
                x_max=0,
                y=self.y_core,
                y_span=self.geom.layers_stack['core'][1],
                z_span=self._Z_SPAN
            )

        core_material = self.geom.layers_stack['core'][0]
        x_min = 0
        self.addstructuregroup(name='grating')
        for i in range(self.geom.n_segments):
            for j in range(len(self.geom.duty_cycle)):
                ff = self.geom.duty_cycle[j]
                x_span = ff * self.geom.pitch
                y_span = self.geom.grating_shape[j]
                y = self.y_core - self.geom.layers_stack['core'][1]/2 + y_span/2
                if x_span==0:
                    continue
                x_max = x_min + x_span
                self.addrect(
                    name=f'segment_{i}_section_{j}',
                    material=core_material,
                    x_min=x_min,
                    x_max=x_max,
                    y=y,
                    y_span=y_span,
                    z_span=self._Z_SPAN
                )
                self.addtogroup('grating')
                x_min = x_max
        x_max += self._BUFFER
        self.addrect(
            name='remaining core',
            material=core_material,
            x_min=x_min,
            x_max=x_max,
            y=self.y_core,
            y_span=y_span,
            z_span=self._Z_SPAN
        )
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
                wavelength_stop=self.source_wl_range[0],
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
        if filename:
            self.save = True
            self._filename = Path(filename)
            self._savepath = self._filename.parent
            if self._filename.suffix != '.lms':
                self._filename = self._filename.with_suffix('.lms').name
            if self._savepath != Path('.'):
                os.makedirs(self._savepath, exist_ok=True)
        else:
            # Create a temporary directory for simulation
            self._tempdir = tempfile.TemporaryDirectory()
            atexit.register(self._cleanup_tempdir)  # Clean up after run
            self._savepath = self._tempdir.name
            self._filename = Path("temp_grating_model.lms")
        super().save(str(self._savepath/self._filename))

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