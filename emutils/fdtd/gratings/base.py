"""Solver independent base class of grating coupler models."""
import logging
import tempfile
import atexit
import numbers
from pathlib import Path

import numpy as np

from .analysis import analyze_spectrum

logger = logging.getLogger(__name__)


class _GratingCouplerBase:
    """
    Solver independent part of grating coupler models: simulation parameters with
    validation, reactive rebuild of the project, save location and results processing.

    Subclasses implement `_create_geometry`, `_add_solver`, `_add_source`, `_add_monitor`,
    `_fetch_transmission` and the `_apply_*` hooks that update existing objects.
    """
    _FILE_SUFFIX = '.lms'
    _LUMERICAL_CLASS = None

    def _check_lumapi(self):
        if self._LUMERICAL_CLASS is object:
            err_msg = "Module 'lumapi' is required to use this functionality!"
            logger.error(err_msg)
            raise ImportError(err_msg)

    def _init_parameters(self, polarization, theta, configuration, source_wl_range):
        self._IS_INITIALIZED = False
        self.polarization = polarization
        self.theta = theta
        self.configuration = configuration
        self.source_wl_range = source_wl_range

    def _on_param_change(self, name, value):
        self._IS_INITIALIZED = False
        self._initialize_objects()

    @property
    def polarization(self):
        return self._polarization

    @polarization.setter
    def polarization(self, value):
        if value.lower() not in ['te', 'tm']:
            raise ValueError(f'Invalid polarization, "TE" or "TM" available, got "{value}"')
        self._polarization = value
        if self._IS_INITIALIZED:
            self._apply_polarization()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        if not (value is None or isinstance(value, numbers.Number)):
            raise TypeError(f'Parameter theta should be None or numeric value, got {type(value)}')
        self._theta = value
        if self._IS_INITIALIZED and self.configuration == 'in':
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

        # source and monitor both depend on configuration
        if self._IS_INITIALIZED:
            self._on_param_change('configuration', value)

    @property
    def source_wl_range(self):
        return self._source_wl_range

    @source_wl_range.setter
    def source_wl_range(self, wl_range):
        if len(wl_range)!=2:
            raise ValueError('Invalid source wavelength range')
        self._source_wl_range = sorted(wl_range)

        if self._IS_INITIALIZED:
            self._apply_wl_range()

    def _apply_polarization(self):
        self._on_param_change('polarization', self.polarization)

    def _apply_wl_range(self):
        self._on_param_change('source_wl_range', self.source_wl_range)

    def _initialize_objects(self):
        self.switchtolayout()
        self.deleteall()

        self._create_geometry()
        self._add_solver()
        self._add_source()
        self._add_monitor()

        self._IS_INITIALIZED = True

    def _setup_save_location(self, save_as):
        self._tempdir = None
        if save_as:
            path = Path(save_as).with_suffix(self._FILE_SUFFIX)
            self._savepath = path.parent
            self._filename = Path(path.name)
            self._savepath.mkdir(parents=True, exist_ok=True)
        else:
            # Create a temporary directory for simulation
            self._tempdir = tempfile.TemporaryDirectory()
            atexit.register(self._cleanup_tempdir)  # Clean up after run
            self._savepath = Path(self._tempdir.name)
            self._filename = Path("temp_grating_model").with_suffix(self._FILE_SUFFIX)
        super().save(str(self._savepath / self._filename))

    def _cleanup_tempdir(self):
        if self._tempdir:
            self._tempdir.cleanup()

    def _fetch_transmission(self):
        """Returns (wavelength, frequency, transmission) of the simulated coupler."""
        raise NotImplementedError

    def _monitor_transmission(self, monitor='flux monitor'):
        try:
            results = self.getresult(monitor, 'T')
        except Exception as e:
            raise RuntimeError(f"Could not fetch result from monitor: {e}")

        if results is None or 'lambda' not in results or 'T' not in results:
            raise RuntimeError("Result does not contain expected data.")

        return (
            np.squeeze(results['lambda']),
            np.squeeze(results['f']),
            np.abs(np.squeeze(results['T'])),
        )

    def get_results(self):
        """
        Retrieve transmission results and compute relevant performance metrics.

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
                    - '1dB': float or None, 1 dB bandwidth (in nm)
                    - '3dB': float or None, 3 dB bandwidth (in nm)
                - 'fit_params': list or None, optimized fit parameters (if fitting succeeded)

        Raises:
            RuntimeError: If the result data is unavailable or improperly structured.
        """
        return analyze_spectrum(*self._fetch_transmission())

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
