import unittest
from emutils import um, nm
from emutils.fdtd.gratings import GratingGeometry, build_rects

BUFFER = 2*um
LAYERS = dict(
    substrate=('Si', 0.5*um),
    box=('SiO2', 2.0*um),
    core=('Si', 220*nm),
    oxide_cladding=('SiO2', 2.0*um),
    cladding=('air', 2.0*um),
)
PITCH = 660*nm
DC = 0.45


def make_geom(**kwargs):
    params = dict(
        layers_stack=LAYERS, grating_shape=(150*nm, 220*nm), pitch=PITCH,
        duty_cycle=DC, n_segments=5, input_wg_length=8*um,
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
        return PITCH + (DC/2 if section == 0 else DC + (1 - DC)/2) * PITCH

    def test_core_only_backward_compatible(self):
        geom = make_geom()
        rects = build_rects(geom, BUFFER)
        self.assertTiled(geom, rects)
        core_bottom, core_top = geom.layer_bounds(BUFFER)['core']

        etched = [r for r in rects if r.name.startswith('core_seg') and r.name.endswith('sec0')]
        self.assertEqual(len(etched), 5)
        for i, r in enumerate(etched):
            self.assertAlmostEqual(r.x_min, i*PITCH)
            self.assertAlmostEqual(r.x_max, i*PITCH + DC*PITCH)
            self.assertAlmostEqual(r.y_min, core_bottom)
            self.assertAlmostEqual(r.y_max - r.y_min, 150*nm)

        fills = [r for r in rects if r.name.startswith('fill')]
        self.assertEqual(len(fills), 5)
        for r in fills:
            self.assertEqual(r.material, 'SiO2')
            self.assertAlmostEqual(r.y_max - r.y_min, 70*nm)

        tail = [r for r in rects if r.name.endswith('tail')]
        self.assertEqual(len(tail), 1)
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
            make_geom(duty_cycle=0.5, grating_shape=(0, 0, 0))

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
        geom.update(grating_shape=(0, 150*nm, 220*nm), duty_cycle=(0.2, 0.3, 0.5))
        self.assertEqual(calls, ['pitch', 'update'])
        self.assertTiled(geom, build_rects(geom, BUFFER))


if __name__ == '__main__':
    unittest.main()
