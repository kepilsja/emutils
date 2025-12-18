import unittest

from emutils.util import dotdict

class TestDotDict(unittest.TestCase):

    def test_attribute_access(self):
        d = dotdict({'a': 1, 'b': 2})
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 2)

    def test_nested_attribute_access(self):
        d = dotdict({'a': {'b': {'c': 3}}})
        self.assertEqual(d.a.b.c, 3)
        self.assertIsInstance(d.a, dotdict)
        self.assertIsInstance(d.a.b, dotdict)

    def test_set_attribute(self):
        d = dotdict()
        d.a = 10
        self.assertEqual(d['a'], 10)
        self.assertEqual(d.a, 10)

    def test_delete_attribute(self):
        d = dotdict({'a': 1})
        del d.a
        self.assertNotIn('a', d)

    def test_missing_attribute_returns_none(self):
        d = dotdict()
        self.assertIsNone(d.missing_key)

    def test_dict_style_access_still_works(self):
        d = dotdict({'a': 1})
        self.assertEqual(d['a'], 1)

    def test_overwrite_existing_value(self):
        d = dotdict({'a': 1})
        d.a = 2
        self.assertEqual(d.a, 2)

    def test_delete_nonexistent_attribute_raises_keyerror(self):
        d = dotdict()
        with self.assertRaises(KeyError):
            del d.a