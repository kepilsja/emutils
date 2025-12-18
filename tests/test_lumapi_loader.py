import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import emutils.lumapi_loader as loader

class TestLumapiLoader(unittest.TestCase):

    def setUp(self):
        self.original_sys_path = list(sys.path)
        self.env_backup = os.environ.copy()

    def tearDown(self):
        sys.path = self.original_sys_path
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_env_variable_sets_path(self):
        fake_path = Path("/fake/lumapi/path")
        os.environ["LUMAPI_PATH"] = str(fake_path)

        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(sys, "path", []):
            loader.setup_lumapi()
            self.assertIn(str(fake_path), sys.path[0])

    def test_already_in_sys_path(self):
        sys.path = ["C:/Program Files/Lumerical/v250/api/python"]

        # Should not modify sys.path or raise anything
        loader.setup_lumapi()
        self.assertIn("C:/Program Files/Lumerical/v250/api/python", sys.path)

    def test_find_lumapi_path_recursive_success(self):
        with mock.patch.object(Path, "rglob") as mock_rglob:
            mock_path = Path("C:/Program Files/Lumerical/v251/api/python")
            mock_rglob.return_value = [mock_path]

            result = loader.find_lumapi_path_recursive([Path("C:/Program Files")])
            self.assertEqual(result, mock_path.resolve())

    def test_find_lumapi_path_recursive_none(self):
        with mock.patch.object(Path, "rglob", return_value=[]):
            result = loader.find_lumapi_path_recursive([Path("C:/Program Files")])
            self.assertIsNone(result)

    def test_setup_lumapi_fallback_search_success(self):
        found_path = Path("C:/Program Files/Lumerical/v251/api/python")

        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(loader, "find_lumapi_path_recursive", return_value=found_path), \
             mock.patch.object(sys, "path", []):
            loader.setup_lumapi()
            self.assertIn(str(found_path), sys.path)

    def test_setup_lumapi_fallback_search_failure(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(loader, "find_lumapi_path_recursive", return_value=None), \
             mock.patch.object(sys, "path", []):
            with self.assertRaises(ImportError):
                loader.setup_lumapi()


if __name__ == "__main__":
    unittest.main()
