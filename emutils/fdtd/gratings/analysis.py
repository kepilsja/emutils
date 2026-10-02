"""Processing of simulated transmission spectra."""
import numpy as np
from scipy.optimize import curve_fit

from ... import nm


def analyze_spectrum(wavelength, frequency, transmission):
    """
    Computes peak transmission and 1 dB / 3 dB bandwidths (in nm) of a transmission
    spectrum using a Gaussian fit. See `get_results` of grating couplers for the
    structure of the returned dictionary.
    """
    wavelength = np.asarray(wavelength)
    frequency = np.asarray(frequency)
    transmission = np.asarray(transmission)

    # Peak transmission info
    max_idx = np.argmax(transmission)
    max_transmission = transmission[max_idx]
    peak_wavelength = wavelength[max_idx]
    peak_frequency = frequency[max_idx]

    def gaussian(x, a, x0, sigma, offset):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

    def find_bandwidth(popt, dB_drop):
        target = max_transmission * 10**(-dB_drop / 10)
        wavelengths = np.linspace(wavelength.min(), wavelength.max(), 3000)
        fitted = gaussian(wavelengths, *popt)
        indices = np.where(fitted >= target)[0]
        if len(indices) >= 2:
            low = wavelengths[indices[0]]/nm
            high = wavelengths[indices[-1]]/nm
            return high - low
        return None

    try:
        popt, _ = curve_fit(gaussian, wavelength, transmission,
                            p0=[max_transmission, peak_wavelength, 0.1e-6, min(transmission)])
        bandwidth_1dB = find_bandwidth(popt, 1)
        bandwidth_3dB = find_bandwidth(popt, 3)
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
