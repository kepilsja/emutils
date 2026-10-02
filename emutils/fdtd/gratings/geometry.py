"""Validated geometry parameters of 2D and focusing (3D) grating couplers."""
from dataclasses import dataclass, field, fields
from typing import Union, Tuple, Dict, Iterable, Callable, Optional
import numbers

import numpy as np

from ... import um, nm


def _is_air(material):
    return material.lower() == 'air'


DEFAULT_LAYERS_STACK = dict(
    substrate=('Si (Silicon) - Palik', 0.5*um),
    box=('SiO2 (Glass) - Palik', 2.0*um),
    core=('Si (Silicon) - Palik', 220*nm),
    oxide_cladding=('SiO2 (Glass) - Palik', 2.0*um),
    cladding=('air', 2.0*um)
)


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

        fill_factor (Union[float, Iterable[float]]):
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
          (e.g. `grating_shape` together with `fill_factor`).
    """
    layers_stack: Dict[str, Tuple[str, float]]
    grating_shape: Iterable[float]
    pitch: float
    fill_factor: Union[float, Iterable[float]]
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

        self._validate_fill_factor()
        if self.core_only:
            self._validate_core_only_shape()
        else:
            self._validate_etch_depths()

    def _validate_fill_factor(self):
        if isinstance(self.fill_factor, numbers.Number):
            if not (0.0 <= self.fill_factor <= 1.0):
                raise ValueError(f"fill_factor value has to be number between 0 and 1, got {self.fill_factor}")
            object.__setattr__(self, 'fill_factor', [1.0 - self.fill_factor, self.fill_factor])
        elif isinstance(self.fill_factor, Iterable):
            object.__setattr__(self, 'fill_factor', list(self.fill_factor))
            total = sum(self.fill_factor)
            if not abs(total - 1.0) < 1e-6:
                raise ValueError(f"fill_factor values must sum to 1, got {total}.")

        else:
            raise TypeError("fill_factor must be a float or an iterable of floats.")

        n_sections = len(list(self.grating_shape))
        if len(self.fill_factor) != n_sections:
            raise ValueError(
                f"fill_factor must match length of grating_shape. "
                f"Got {len(self.fill_factor)} and {n_sections}."
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
        """
        Returns {layer_name: (y_min, y_max)}, bottom and top layers are extended by `buffer`.
        The vertical axis is y in the 2D model and z in the 3D model.
        """
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


def confocal_ellipticity(n_eff, n_clad, theta):
    """
    Ellipticity of confocal grating lines of a focusing grating coupler,
    kappa = n_clad*sin(theta)/n_eff, where theta (deg) is the fiber angle in the cladding
    (n_clad*sin(theta) is conserved, so the angle in air with n_clad=1 can be used as well).
    """
    return n_clad * np.sin(np.radians(theta)) / n_eff


@dataclass
class FocusingGratingGeometry(GratingGeometry):
    """
    Geometry of a focusing grating coupler. The vertical stack and the grating profile
    are the ones of `GratingGeometry`, the profile is defined along the symmetry axis
    of the device (phi=0) and measured from the first grating line.

    In-plane layout: the focal point is at the origin, light propagates along +x.
    An input waveguide of width `wg_width` joins a sector-shaped taper of full opening
    angle `taper_angle` that ends at distance `taper_length` from the focal point,
    where the grating starts. Grating lines are confocal ellipses:

        r(phi) = r_axis * (1 - kappa) / (1 - kappa*cos(phi)),

    where kappa is the `ellipticity` (0 gives circular arcs, see `confocal_ellipticity`).

    Additional attributes:
        taper_length (float): Distance from the focal point to the first grating line.
        taper_angle (float): Full opening angle of the taper/grating sector in degrees.
        wg_width (float): Width of the input waveguide.
        ellipticity (float): kappa in [0, 1).
        trench_depth (float, optional): Outside the device footprint the stack is etched
            from the top of the etch layer down by `trench_depth` and filled with the
            fill material. Defaults to etching down to the core bottom.
    """
    taper_length: float = 15*um
    taper_angle: float = 30.0
    wg_width: float = 500*nm
    ellipticity: float = 0.0
    trench_depth: Optional[float] = None

    def _validate(self):
        super()._validate()

        for name in ('taper_length', 'wg_width'):
            value = getattr(self, name)
            if not isinstance(value, numbers.Number) or value <= 0:
                raise ValueError(f"{name} must be a positive number.")

        if not isinstance(self.taper_angle, numbers.Number) or not (0 < self.taper_angle < 180):
            raise ValueError(f"taper_angle must be between 0 and 180 degrees, got {self.taper_angle}.")

        if not isinstance(self.ellipticity, numbers.Number) or not (0 <= self.ellipticity < 1):
            raise ValueError(f"ellipticity must be in range [0, 1), got {self.ellipticity}.")

        half_angle = np.radians(self.taper_angle) / 2
        if self.radius(0.0, half_angle) * np.cos(half_angle) <= self.taper_join_x():
            raise ValueError(
                "Taper is too short or too narrow for the waveguide width: increase "
                "taper_length or taper_angle, or decrease wg_width."
            )

        if self.trench_depth is not None and not isinstance(self.trench_depth, numbers.Number):
            raise TypeError("trench_depth must be a number or None.")
        y_top = self.layer_bounds()[self._etch_from_layer()][1]
        depth = self.trench_depth_value()
        if not (0 < depth <= y_top):
            raise ValueError(
                f"trench_depth must be positive and not exceed the stack thickness below "
                f"the etch top ({y_top}), got {depth}."
            )

    def trench_depth_value(self):
        if self.trench_depth is not None:
            return self.trench_depth
        y_top = self.layer_bounds()[self._etch_from_layer()][1]
        return y_top - self.layer_bounds()['core'][0]

    def radius(self, x_axis, phi):
        """Distance from the focal point to the grating line crossing the axis at grating coordinate `x_axis`."""
        kappa = self.ellipticity
        return (self.taper_length + x_axis) * (1 - kappa) / (1 - kappa * np.cos(phi))

    def taper_join_x(self):
        """x position where the straight waveguide joins the taper sector."""
        return self.wg_width / (2 * np.tan(np.radians(self.taper_angle) / 2))

    def _phis(self, n_phi):
        half_angle = np.radians(self.taper_angle) / 2
        return np.linspace(-half_angle, half_angle, n_phi)

    def _curve(self, x_axis, n_phi):
        phi = self._phis(n_phi)
        r = self.radius(x_axis, phi)
        return np.column_stack((r * np.cos(phi), r * np.sin(phi)))

    def sector_polygon(self, x0, x1, n_phi=64):
        """Vertices (N, 2) of the sector region between grating lines crossing the axis at x0 < x1."""
        return np.vstack((self._curve(x1, n_phi), self._curve(x0, n_phi)[::-1]))

    def taper_polygon(self, n_phi=64):
        """Vertices (N, 2) of the taper between the waveguide joint and the first grating line."""
        x_join = self.taper_join_x()
        return np.vstack((
            [[x_join, -self.wg_width / 2]],
            self._curve(0.0, n_phi),
            [[x_join, self.wg_width / 2]],
        ))

    def sector_half_width(self, x_axis, n_phi=64):
        """Maximal |y| of the grating line crossing the axis at `x_axis`."""
        return np.abs(self._curve(x_axis, n_phi)[:, 1]).max()
