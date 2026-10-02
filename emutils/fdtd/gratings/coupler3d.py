"""3D focusing grating coupler simulated in Lumerical FDTD."""
from typing import Union
from pathlib import Path

import numpy as np

from ... import um, nm
from ._lumapi import LumerFDTD
from .base import _GratingCouplerBase
from .cross_section import profile_intervals, layer_runs
from .geometry import FocusingGratingGeometry, DEFAULT_LAYERS_STACK, _is_air


class GratingCoupler3D(_GratingCouplerBase, LumerFDTD):
    """Automates the creation and simulation of a 3D focusing grating coupler in Lumerical FDTD.

    The vertical stack and the grating profile are defined exactly as in
    `GratingCoupler2D`; the profile lies along the symmetry axis of the device and
    the grating lines are confocal ellipses (see `FocusingGratingGeometry`).

    Coordinates: the focal point of the grating is at the origin, light propagates
    along +x, the stack is along z (z=0 at the bottom of the substrate layer extended
    by the buffer) and y=0 is the symmetry plane. Outside the device footprint
    (waveguide + taper + grating sector) the stack is etched down by `trench_depth`
    and filled with the fill material.

    Args:
        layers_stack, grating_shape, pitch, fill_factor, n_segments, input_wg_length,
        core_only, etch_from, fill_material, polarization, configuration, theta,
        source_wl_range, save_as:
            See `GratingCoupler2D`. `input_wg_length` is the length of the straight
            waveguide before the taper (defaults to 5*um).
        taper_length (float, optional): Distance from the focal point to the first
            grating line. Defaults to 15*um.
        taper_angle (float, optional): Full opening angle of the taper in degrees.
            Defaults to 30.
        wg_width (float, optional): Width of the input waveguide. Defaults to 500*nm.
        ellipticity (float, optional): Ellipticity of the confocal grating lines,
            see `confocal_ellipticity`. Defaults to 0 (circular arcs).
        trench_depth (float, optional): Etch depth outside the device footprint.
            Defaults to etching down to the core bottom.
        fiber_x (float, optional): x position of the fiber (Gaussian beam) center for
            the 'in' configuration. Defaults to taper_length + 4.5*um.
        use_symmetry (bool, optional): Use symmetry boundary at y=0 to halve the
            simulation volume. Defaults to True.
        mesh_accuracy (int, optional): FDTD mesh accuracy. Defaults to 2.
        **kwargs:
            Additional keyword arguments passed to the parent `LumerFDTD` class
            (e.g. `filename` of the project to open, `hide`).

    Attributes:
        geom (FocusingGratingGeometry): Validated geometric parameters. Changing
            attributes on this object will trigger a simulation re-build.
        solver, source, polarization, theta, configuration, source_wl_range:
            See `GratingCoupler2D`.

    Notes:
        For the 'in' configuration the transmission is the power coupled into the
        fundamental waveguide mode (mode expansion monitor). For the 'out'
        configuration it is the total power flowing through the plane above the
        grating (no overlap with the fiber mode).
    """
    _FILE_SUFFIX = '.fsp'
    _LUMERICAL_CLASS = LumerFDTD

    def __init__(self,
            layers_stack=DEFAULT_LAYERS_STACK,
            grating_shape=(150*nm, 220*nm),
            pitch=660*nm,
            fill_factor=0.45,
            n_segments=25,
            input_wg_length=5*um,
            core_only=True,
            etch_from=None,
            fill_material=None,
            taper_length=15*um,
            taper_angle=30.0,
            wg_width=500*nm,
            ellipticity=0.0,
            trench_depth=None,
            polarization='te',
            configuration='in',
            theta=20,
            source_wl_range=(1450*nm, 1650*nm),
            fiber_x=None,
            use_symmetry=True,
            mesh_accuracy=2,
            save_as:Union[str, Path]=None,
            **kwargs
        ):
        self._check_lumapi()
        super().__init__(**kwargs)

        self._BUFFER = 2*um
        self._MARGIN = 1*um
        self._FREQ_POINTS = 81
        self._BEAM_WAIST_RADIUS = 5.2*um
        self._FIBER_OFFSET = 4.5*um
        self._MESH_DZ = 20*nm
        self._N_PHI = 64

        self._fiber_x = fiber_x
        self.use_symmetry = use_symmetry
        self.mesh_accuracy = mesh_accuracy
        self._init_parameters(polarization, theta, configuration, source_wl_range)

        self.geom = FocusingGratingGeometry(
            dict(layers_stack), grating_shape, pitch, fill_factor, n_segments, input_wg_length,
            core_only=core_only, etch_from=etch_from, fill_material=fill_material,
            taper_length=taper_length, taper_angle=taper_angle, wg_width=wg_width,
            ellipticity=ellipticity, trench_depth=trench_depth,
            _callback=self._on_param_change,
        )

        self._setup_save_location(save_as)
        self._initialize_objects()

    @property
    def grating_length(self):
        return self.geom.n_segments * self.geom.pitch

    @property
    def x_sim_min(self):
        return self.geom.taper_join_x() - self.geom.input_wg_length

    @property
    def x_sim_max(self):
        return self.geom.taper_length + self.grating_length + self._MARGIN

    @property
    def y_sim_half(self):
        return self.geom.sector_half_width(self.grating_length, self._N_PHI) + self._MARGIN

    @property
    def sz(self):
        return sum(height for _, height in self.geom.layers_stack.values()) + 2*self._BUFFER

    @property
    def z_core(self):
        z_min, z_max = self.geom.layer_bounds(self._BUFFER)['core']
        return (z_min + z_max) / 2

    @property
    def fiber_x(self):
        if self._fiber_x is None:
            return self.geom.taper_length + self._FIBER_OFFSET
        return self._fiber_x

    @fiber_x.setter
    def fiber_x(self, value):
        self._fiber_x = value
        if self._IS_INITIALIZED and self.configuration == 'in':
            self.source.x = self.fiber_x

    @property
    def source_loc(self):
        return self._source_loc

    @source_loc.setter
    def source_loc(self, loc):
        x, y, z = loc
        self._source_loc = (x, y, z)
        if self._IS_INITIALIZED:
            self.source.x = x
            self.source.y = y
            self.source.z = z

    @property
    def monitor_loc(self):
        return self._monitor_loc

    @monitor_loc.setter
    def monitor_loc(self, loc):
        x, y, z = loc
        self._monitor_loc = (x, y, z)
        if self._IS_INITIALIZED:
            self._place_monitors()

    def _place_monitors(self):
        x, y, z = self._monitor_loc
        monitors = ['flux monitor'] + (['mode expansion'] if self.configuration == 'in' else [])
        for name in monitors:
            self.select(name)
            self.set('x', x)
            self.set('y', y)
            self.set('z', z)

    def _apply_wl_range(self):
        self.setglobalsource('wavelength start', self.source_wl_range[0])
        self.setglobalsource('wavelength stop', self.source_wl_range[1])

    def _etch_zone(self):
        """(z_bottom, z_top) of the trench etched outside the device footprint."""
        z_top, _, _ = self.geom.section_surfaces(self._BUFFER)
        return z_top - self.geom.trench_depth_value(), z_top

    def _set_mesh_order(self, order):
        self.set('override mesh order from material database', 1)
        self.set('mesh order', order)

    def _add_slab(self, name, material, z_min, z_max, mesh_order):
        x_min = self.x_sim_min - self._BUFFER
        x_max = self.x_sim_max + self._BUFFER
        obj = self.addrect(
            name=name,
            x=(x_min + x_max) / 2, x_span=x_max - x_min,
            y=0, y_span=2 * (self.y_sim_half + self._BUFFER),
            z=(z_min + z_max) / 2, z_span=z_max - z_min,
        )
        obj.material = material
        self._set_mesh_order(mesh_order)

    def _add_prism(self, name, material, vertices, z_min, z_max):
        obj = self.addpoly(
            name=name, x=0, y=0,
            z=(z_min + z_max) / 2, z_span=z_max - z_min,
            vertices=vertices,
        )
        obj.material = material
        self._set_mesh_order(1)
        self.addtogroup('grating')

    def _create_geometry(self):
        eps = 1e-15
        geom = self.geom
        z_bottom, z_top = self._etch_zone()
        _, fill, _ = geom.section_surfaces(self._BUFFER)

        # background: full layers and the trench outside the device footprint
        for name, (z_min, z_max) in geom.layer_bounds(self._BUFFER).items():
            material = geom.layers_stack[name][0]
            if not _is_air(material):
                self._add_slab(f'layer_{name}', material, z_min, z_max, mesh_order=3)
        self._add_slab('trench', 'etch' if _is_air(fill) else fill, z_bottom, z_top, mesh_order=2)

        # device footprint: unetched waveguide and taper in the etch zone layers
        self.addstructuregroup(name='grating')
        x_join = geom.taper_join_x()
        x_wg_min = self.x_sim_min - self._BUFFER
        taper = geom.taper_polygon(self._N_PHI)
        for name, (z_min, z_max) in geom.layer_bounds(self._BUFFER).items():
            material = geom.layers_stack[name][0]
            if _is_air(material) or z_max <= z_bottom + eps or z_min >= z_top - eps:
                continue
            obj = self.addrect(
                name=f'wg_{name}',
                x=(x_wg_min + x_join) / 2, x_span=x_join - x_wg_min,
                y=0, y_span=geom.wg_width,
                z=(z_min + z_max) / 2, z_span=z_max - z_min,
            )
            obj.material = material
            self._set_mesh_order(1)
            self.addtogroup('grating')
            self._add_prism(f'taper_{name}', material, taper, z_min, z_max)

        # grating sectors: the 2D cross-section mapped onto confocal ellipses
        intervals = profile_intervals(geom, self._BUFFER, include_input=False)
        for rect in layer_runs(geom, intervals, self._BUFFER):
            if rect.y_max <= z_bottom + eps or rect.y_min >= z_top - eps:
                continue
            vertices = geom.sector_polygon(rect.x_min, rect.x_max, self._N_PHI)
            self._add_prism(rect.name, rect.material, vertices, rect.y_min, rect.y_max)

    def _add_solver(self):
        z_bottom, z_top = self._etch_zone()
        self.solver = self.addfdtd(
            dimension='3D',
            x=(self.x_sim_min + self.x_sim_max) / 2,
            x_span=self.x_sim_max - self.x_sim_min,
            y=0,
            y_span=2 * self.y_sim_half,
            z=self.sz / 2,
            z_span=self.sz - 2 * self._BUFFER,
            mesh_accuracy=self.mesh_accuracy,
            simulation_time=15000e-15
        )
        if self.use_symmetry:
            # TE mode: Ey is even in y, i.e. E is anti-symmetric as a vector
            self.solver.y_min_bc = 'Anti-Symmetric' if self.polarization.lower() == 'te' else 'Symmetric'

        self.addmesh(
            name='grating mesh',
            x=(self.x_sim_min + self.x_sim_max) / 2,
            x_span=self.x_sim_max - self.x_sim_min,
            y=0,
            y_span=2 * self.y_sim_half,
            z=(z_bottom + z_top) / 2,
            z_span=z_top - z_bottom,
            override_x_mesh=0,
            override_y_mesh=0,
            override_z_mesh=1,
            dz=self._MESH_DZ,
        )

    def _add_source(self):
        if self.configuration == 'in':
            self.source = self.addgaussian(
                name='source',
                injection_axis='z-axis',
                direction='backward',
                x=self.fiber_x,
                x_span=self.x_sim_max - self.x_sim_min,
                y=0,
                y_span=2 * self.y_sim_half,
                z=self.sz - 1.1*self._BUFFER,
                angle_theta=-self._theta,
                polarization_angle=90 if self.polarization.lower() == 'te' else 0,
                waist_radius_w0=self._BEAM_WAIST_RADIUS,
            )
        elif self.configuration == 'out':
            self.source = self.addmode(
                name='source',
                injection_axis='x-axis',
                direction='forward',
                x=self.x_sim_min + 0.1*self.geom.input_wg_length,
                y=0,
                y_span=self.geom.wg_width + 2.5*um,
                z=self.z_core,
                z_span=7*self.geom.layers_stack['core'][1],
                mode_selection=f'fundamental {self.polarization.upper()} mode',
            )
        self._source_loc = (self.source.x, self.source.y, self.source.z)
        self._apply_wl_range()

    def _add_monitor(self):
        self.addpower(name='flux monitor')
        self.set('override global monitor settings', 1)
        self.set('frequency points', self._FREQ_POINTS)

        if self.configuration == 'in':
            loc = (self.x_sim_min + 0.1*self.geom.input_wg_length, 0, self.z_core)
            spans = dict(y_span=self.geom.wg_width + 2.5*um, z_span=7*self.geom.layers_stack['core'][1])
            self.set('monitor type', '2D X-normal')
            self.set('y span', spans['y_span'])
            self.set('z span', spans['z_span'])

            self.addmodeexpansion(name='mode expansion')
            self.set('monitor type', '2D X-normal')
            self.set('y span', spans['y_span'])
            self.set('z span', spans['z_span'])
            self.set('mode selection', f'fundamental {self.polarization.upper()} mode')
            self.set('override global monitor settings', 1)
            self.set('frequency points', self._FREQ_POINTS)
            self.setexpansion('waveguide', 'flux monitor')

        else:
            loc = (
                (self.x_sim_min + self.x_sim_max) / 2, 0,
                self.sz - self._BUFFER - 0.1*self.geom.layers_stack['cladding'][1]
            )
            self.set('monitor type', '2D Z-normal')
            self.set('x span', self.x_sim_max - self.x_sim_min)
            self.set('y span', 2 * self.y_sim_half)

        self._monitor_loc = loc
        self._place_monitors()

    def _fetch_transmission(self):
        if self.configuration == 'out':
            return self._monitor_transmission('flux monitor')

        try:
            results = self.getresult('mode expansion', 'expansion for waveguide')
        except Exception as e:
            raise RuntimeError(f"Could not fetch result from mode expansion monitor: {e}")
        if results is None or 'lambda' not in results or 'T_backward' not in results:
            raise RuntimeError("Result does not contain expected data.")

        # light coupled from the fiber travels towards -x in the waveguide
        return (
            np.squeeze(results['lambda']),
            np.squeeze(results['f']),
            np.abs(np.squeeze(results['T_backward'])),
        )
