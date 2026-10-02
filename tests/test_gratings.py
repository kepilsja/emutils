import unittest
import numpy as np
from emutils import um, nm
from emutils.fdtd import gratings
from emutils.fdtd.gratings import (
    GratingGeometry, FocusingGratingGeometry, build_rects, profile_intervals, layer_runs,
    confocal_ellipticity, analyze_spectrum,
)

BUFFER = 2*um
LAYERS = dict(
    substrate=('Si', 0.5*um),
    box=('SiO2', 2.0*um),
    core=('Si', 220*nm),
    oxide_cladding=('SiO2', 2.0*um),
    cladding=('air', 2.0*um),
)
PITCH = 660*nm
FF = 0.45


def make_geom(**kwargs):
    params = dict(
        layers_stack=LAYERS, grating_shape=(150*nm, 220*nm), pitch=PITCH,
        fill_factor=FF, n_segments=5, input_wg_length=8*um,
    )
    params.update(kwargs)
    return GratingGeometry(**params)


def column(rects, x):
    """Rects crossing vertical line at x, sorted bottom to top."""
    return sorted((r for r in rects if r.x_min < x < r.x_max), key=lambda r: r.y_min)


def material_at(rects, x, y):
    hits = [r.material for r in rects if r.x_min < x < r.x_max and r.y_min < y < r.y_max]
    assert len(hits) <= 1, f"Overlapping rects at ({x}, {y}): {hits}"
    return hits[0] if hits else 'air'


class TestGratingGeometry(unittest.TestCase):

    def assertTiled(self, geom, rects):
        """Every column has non-overlapping rects covering stack from bottom up to the top of non-air layers."""
        bounds = geom.layer_bounds(BUFFER)
        x_min = min(r.x_min for r in rects)
        x_max = max(r.x_max for r in rects)
        edges = sorted({r.x_min for r in rects} | {r.x_max for r in rects})
        for x0, x1 in zip(edges, edges[1:]):
            x = (x0 + x1) / 2
            y = 0.0
            for r in column(rects, x):
                self.assertAlmostEqual(r.y_min, y, delta=1e-15, msg=f"gap/overlap at x={x}")
                y = r.y_max
        self.assertAlmostEqual(x_min, -BUFFER - geom.input_wg_length)
        self.assertAlmostEqual(x_max, geom.n_segments*geom.pitch + BUFFER)
        self.assertAlmostEqual(y, bounds['oxide_cladding'][1])

    def sample_x(self, section):
        """x in the middle of given section of the 2nd segment"""
        etched = 1 - FF
        return PITCH + (etched/2 if section == 0 else etched + FF/2) * PITCH

    def test_core_only_default(self):
        geom = make_geom()
        rects = build_rects(geom, BUFFER)
        self.assertTiled(geom, rects)
        core_bottom, core_top = geom.layer_bounds(BUFFER)['core']

        etched = [r for r in rects if r.name.startswith('core_seg') and r.name.endswith('sec0')]
        self.assertEqual(len(etched), 5)
        for i, r in enumerate(etched):
            self.assertAlmostEqual(r.x_min, i*PITCH)
            self.assertAlmostEqual(r.x_max, i*PITCH + (1 - FF)*PITCH)
            self.assertAlmostEqual(r.y_min, core_bottom)
            self.assertAlmostEqual(r.y_max - r.y_min, 150*nm)

        # unetched teeth have width FF*PITCH, the last one is merged with the tail
        teeth = [r for r in rects if r.name.startswith('core_seg') and r.name.endswith('sec1')]
        self.assertEqual(len(teeth), 4)
        for i, r in enumerate(teeth):
            self.assertAlmostEqual(r.x_min, i*PITCH + (1 - FF)*PITCH)
            self.assertAlmostEqual(r.x_max - r.x_min, FF*PITCH)
            self.assertAlmostEqual(r.y_max - r.y_min, 220*nm)

        fills = [r for r in rects if r.name.startswith('fill')]
        self.assertEqual(len(fills), 5)
        for r in fills:
            self.assertEqual(r.material, 'SiO2')
            self.assertAlmostEqual(r.y_max - r.y_min, 70*nm)

        tail = [r for r in rects if r.name.endswith('tail')]
        self.assertEqual(len(tail), 1)
        self.assertEqual(tail[0].name, 'core_seg4_sec1-tail')
        self.assertAlmostEqual(tail[0].x_min, 4*PITCH + (1 - FF)*PITCH)
        self.assertAlmostEqual(tail[0].y_max - tail[0].y_min, 220*nm)

    def test_core_only_etched_last_section(self):
        # previously 'remaining core' got the height of the last section
        geom = make_geom(grating_shape=(220*nm, 150*nm))
        rects = build_rects(geom, BUFFER)
        self.assertTiled(geom, rects)
        tail = next(r for r in rects if r.name == 'core_tail')
        self.assertAlmostEqual(tail.y_max - tail.y_min, 220*nm)

    def test_partial_top_cladding_etch(self):
        geom = make_geom(core_only=False, etch_from='oxide_cladding', grating_shape=(1*um, 0))
        rects = build_rects(geom, BUFFER)
        self.assertTiled(geom, rects)
        self.assertFalse(any(r.name.startswith('fill') for r in rects))  # air fill
        self.assertTrue(any(r.name == 'core' for r in rects))  # core untouched
        top = geom.layer_bounds(BUFFER)['oxide_cladding'][1]
        self.assertEqual(material_at(rects, self.sample_x(0), top - 0.5*um), 'air')
        self.assertEqual(material_at(rects, self.sample_x(0), top - 1.5*um), 'SiO2')
        self.assertEqual(material_at(rects, self.sample_x(1), top - 0.5*um), 'SiO2')

    def test_through_cladding_partial_core(self):
        geom = make_geom(core_only=False, grating_shape=(2.07*um, 0))
        rects = build_rects(geom, BUFFER)
        self.assertTiled(geom, rects)
        core_bottom, core_top = geom.layer_bounds(BUFFER)['core']
        x = self.sample_x(0)
        self.assertEqual(material_at(rects, x, core_bottom + 100*nm), 'Si')
        self.assertEqual(material_at(rects, x, core_bottom + 200*nm), 'air')
        self.assertEqual(material_at(rects, x, core_top + 1*um), 'air')
        self.assertEqual(material_at(rects, self.sample_x(1), core_top + 1*um), 'SiO2')

    def test_deep_etch_into_box(self):
        geom = make_geom(core_only=False, grating_shape=(2.22*um + 0.5*um, 0), fill_material='SiN')
        rects = build_rects(geom, BUFFER)
        self.assertTiled(geom, rects)
        box_bottom, box_top = geom.layer_bounds(BUFFER)['box']
        core_bottom, _ = geom.layer_bounds(BUFFER)['core']
        x = self.sample_x(0)
        self.assertEqual(material_at(rects, x, core_bottom + 100*nm), 'SiN')
        self.assertEqual(material_at(rects, x, box_top - 0.25*um), 'SiN')
        self.assertEqual(material_at(rects, x, box_top - 1*um), 'SiO2')
        self.assertEqual(material_at(rects, self.sample_x(1), core_bottom + 100*nm), 'Si')

    def test_validation(self):
        with self.assertRaises(ValueError):
            make_geom(grating_shape=(150*nm, 300*nm))
        with self.assertRaises(ValueError):
            make_geom(core_only=False, etch_from='nope', grating_shape=(0, 0))
        with self.assertRaises(ValueError):
            make_geom(core_only=False, grating_shape=(10*um, 0))
        with self.assertRaises(ValueError):
            make_geom(fill_factor=0.5, grating_shape=(0, 0, 0))

    def test_setattr_validates_and_reverts(self):
        geom = make_geom()
        with self.assertRaises(ValueError):
            geom.grating_shape = (150*nm, 300*nm)
        self.assertEqual(tuple(geom.grating_shape), (150*nm, 220*nm))

    def test_update_calls_callback_once(self):
        calls = []
        geom = make_geom(_callback=lambda k, v: calls.append(k))
        geom.pitch = 700*nm
        self.assertEqual(calls, ['pitch'])
        geom.update(grating_shape=(0, 150*nm, 220*nm), fill_factor=(0.2, 0.3, 0.5))
        self.assertEqual(calls, ['pitch', 'update'])
        self.assertTiled(geom, build_rects(geom, BUFFER))


def make_focusing_geom(**kwargs):
    params = dict(
        layers_stack=LAYERS, grating_shape=(150*nm, 220*nm), pitch=PITCH,
        fill_factor=FF, n_segments=5, input_wg_length=5*um,
    )
    params.update(kwargs)
    return FocusingGratingGeometry(**params)


class TestProfileSplit(unittest.TestCase):

    def test_intervals_without_input(self):
        geom = make_geom()
        intervals = profile_intervals(geom, BUFFER, include_input=False)
        self.assertEqual(intervals[0][1], 0.0)
        self.assertNotIn('input', [label for label, *_ in intervals])

    def test_layer_runs_match_build_rects_in_grating(self):
        geom = make_geom()
        full = build_rects(geom, BUFFER)
        grating_only = layer_runs(geom, profile_intervals(geom, BUFFER, include_input=False), BUFFER)
        self.assertEqual(
            {r for r in full if r.x_min >= 0},
            {r for r in grating_only if r.in_grating},
        )


class TestFocusingGratingGeometry(unittest.TestCase):

    def test_defaults(self):
        geom = make_focusing_geom()
        self.assertAlmostEqual(geom.trench_depth_value(), 220*nm)
        self.assertAlmostEqual(geom.taper_join_x(), 250*nm / np.tan(np.radians(15)))

    def test_validation(self):
        with self.assertRaises(ValueError):
            make_focusing_geom(taper_angle=0)
        with self.assertRaises(ValueError):
            make_focusing_geom(taper_angle=180)
        with self.assertRaises(ValueError):
            make_focusing_geom(ellipticity=1.0)
        with self.assertRaises(ValueError):
            make_focusing_geom(taper_length=1*um, taper_angle=2, wg_width=1*um)
        with self.assertRaises(ValueError):
            make_focusing_geom(trench_depth=10*um)
        with self.assertRaises(ValueError):
            make_focusing_geom(grating_shape=(150*nm, 300*nm))  # inherited validation

    def test_update_and_callback(self):
        calls = []
        geom = make_focusing_geom(_callback=lambda k, v: calls.append(k))
        geom.taper_angle = 40
        self.assertEqual(calls, ['taper_angle'])
        with self.assertRaises(ValueError):
            geom.ellipticity = -0.1
        self.assertEqual(geom.ellipticity, 0.0)
        self.assertEqual(calls, ['taper_angle'])

    def test_radius(self):
        phi = np.linspace(-0.3, 0.3, 7)
        circular = make_focusing_geom()
        np.testing.assert_allclose(circular.radius(2*um, phi), 17*um)

        kappa = confocal_ellipticity(n_eff=2.8, n_clad=1.0, theta=20)
        geom = make_focusing_geom(ellipticity=kappa)
        self.assertAlmostEqual(geom.radius(2*um, 0.0), 17*um)
        # confocal (phase matching) condition: n_eff*r - n_clad*sin(theta)*x is constant on a line
        r = geom.radius(2*um, phi)
        phase = 2.8*r - np.sin(np.radians(20)) * r * np.cos(phi)
        np.testing.assert_allclose(phase, phase[0])

    def test_sector_polygon(self):
        n_phi = 32
        geom = make_focusing_geom(ellipticity=0.15)
        vertices = geom.sector_polygon(1*um, 1.3*um, n_phi)
        self.assertEqual(vertices.shape, (2*n_phi, 2))
        angles = np.arctan2(vertices[:, 1], vertices[:, 0])
        half_angle = np.radians(geom.taper_angle) / 2
        self.assertTrue(np.all(np.abs(angles) <= half_angle + 1e-12))
        outer = np.hypot(*vertices[:n_phi].T)
        inner = np.hypot(*vertices[n_phi:][::-1].T)
        self.assertTrue(np.all(outer > inner))
        # outer curve goes with increasing angle, inner one back
        self.assertTrue(np.all(np.diff(angles[:n_phi]) > 0))
        self.assertTrue(np.all(np.diff(angles[n_phi:]) < 0))

    def test_taper_polygon(self):
        geom = make_focusing_geom()
        vertices = geom.taper_polygon(16)
        np.testing.assert_allclose(vertices[0], [geom.taper_join_x(), -geom.wg_width/2])
        np.testing.assert_allclose(vertices[-1], [geom.taper_join_x(), geom.wg_width/2])
        np.testing.assert_allclose(np.hypot(*vertices[1:-1].T), geom.taper_length)


class TestAnalyzeSpectrum(unittest.TestCase):

    def test_gaussian_spectrum(self):
        wavelength = np.linspace(1450*nm, 1650*nm, 81)
        frequency = 299792458 / wavelength
        sigma = 20*nm
        transmission = 0.6 * np.exp(-(wavelength - 1550*nm)**2 / (2*sigma**2))
        res = analyze_spectrum(wavelength, frequency, transmission)
        self.assertAlmostEqual(res['peak']['max_transmission'], 0.6)
        self.assertAlmostEqual(res['peak']['wavelength'], 1550*nm)
        self.assertAlmostEqual(res['bandwidth']['3dB'], 2*np.sqrt(2*np.log(10**0.3))*sigma/nm, delta=0.5)


class TestDeprecatedAlias(unittest.TestCase):

    def test_grating_coupler_alias(self):
        with self.assertWarns(DeprecationWarning):
            cls = gratings.GratingCoupler
        self.assertIs(cls, gratings.GratingCoupler2D)


if __name__ == '__main__':
    unittest.main()
