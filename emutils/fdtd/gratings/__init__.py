"""
Grating coupler models.

Modules:
    geometry:      GratingGeometry, FocusingGratingGeometry - validated geometry parameters
    cross_section: Rect, build_rects - cross-section as non-overlapping rectangles
    analysis:      analyze_spectrum - peak and bandwidth of transmission spectra
    base:          solver independent base class of the couplers
    coupler2d:     GratingCoupler2D - varFDTD (Lumerical MODE)
    coupler3d:     GratingCoupler3D - focusing grating coupler in 3D FDTD
"""
import warnings

from .geometry import (
    GratingGeometry, FocusingGratingGeometry, confocal_ellipticity, DEFAULT_LAYERS_STACK,
)
from .cross_section import Rect, profile_intervals, layer_runs, build_rects
from .analysis import analyze_spectrum
from .coupler2d import GratingCoupler2D, POLARIZATION_MAP
from .coupler3d import GratingCoupler3D


def __getattr__(name):
    if name == 'GratingCoupler':
        warnings.warn(
            "GratingCoupler is deprecated, use GratingCoupler2D instead.",
            DeprecationWarning, stacklevel=2,
        )
        return GratingCoupler2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
