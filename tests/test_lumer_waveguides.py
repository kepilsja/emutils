import unittest
from emutils.lumer import waveguides as wg
from emutils import um, nm
from emutils.util import dotdict
from emutils.materials import *

DEFAULTS = dotdict(
    substrate_width = 30*um,
    substrate_height = 20*um,
    core_width = 2*um,
    core_height = 1*um,
    etch_depth = 500*nm,
    materials = dotdict(
        core = ge_palik,
        substrate = si_palik,
        film = ge_palik
    ),
)

try:
    import lumapi
    HAS_LUMAPI = True
except ImportError:
    HAS_LUMAPI = False

@unittest.skipUnless(HAS_LUMAPI, "lumapi is not installed")
class TestLumerWaveguides(unittest.TestCase):
    def setUp(self):
        try:
            self.model = wg.Rib(
                core_width=DEFAULTS.core_width,
                core_height=DEFAULTS.core_height,
                etch_depth=DEFAULTS.etch_depth,
                materials=DEFAULTS.materials,
                hide=True
            )
        except Exception as e:
            self.skipTest(f"Could not create a Lumerical model, error ocured:\n{e}")

    def tearDown(self):
        self.model.close()

    def test_rib_model_geometry(self):
        component = self.model.geometry.core
        self.assertEqual(component.x_span, DEFAULTS.core_width)
        self.assertEqual(component.x, 0)
        self.assertEqual(component.y_span, DEFAULTS.core_height)
        self.assertEqual(component.y, DEFAULTS.core_height/2)
        self.assertEqual(component.material, DEFAULTS.materials.core)
        
        component =self.model.geometry.substrate
        self.assertEqual(component.x_span, DEFAULTS.substrate_width)
        self.assertEqual(component.x, 0)
        self.assertEqual(component.y_span, DEFAULTS.substrate_height)
        self.assertEqual(component.y, -DEFAULTS.substrate_height/2)
        self.assertEqual(component.material, DEFAULTS.materials.substrate)

        component =self.model.geometry.film
        self.assertEqual(component.x_span, DEFAULTS.substrate_width)
        self.assertEqual(component.x, 0)
        self.assertEqual(component.y_span, DEFAULTS.core_height-DEFAULTS.etch_depth)
        self.assertEqual(component.y, (DEFAULTS.core_height-DEFAULTS.etch_depth)/2)
        self.assertEqual(component.material, DEFAULTS.materials.core)
