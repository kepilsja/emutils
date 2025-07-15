import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

from emutils import farfield
from emutils.farfield import fit_gaussian

class TestFarfieldDataLoading(unittest.TestCase):
    def test_get_lumer_datafile_dims(self):
        file_path = Path('./tests/test_data/lumer_datafile_dims.txt')
        self.assertEqual(
            farfield._get_lumer_datafile_dims(file_path),
            [233, 41, 1111]
        )

    def test_standard_txt_loading(self):
        file_path = Path('./tests/test_data/farfield_standard_text_data_export_temp.txt')
        ux, uy, data = farfield.load_data(file_path)
        self.assertEqual(ux.shape, (2,))
        self.assertEqual(uy.shape, (3,))
        self.assertEqual(data[0,0], np.complex128(5.488135039273e-01, 4.375872112627e-01))

    def test_matfile_loading(self):
        txt_file_path = Path('./tests/test_data/farfield_test_standard.txt')
        mat_file_path = Path('./tests/test_data/farfield_test.mat')

        ux_txt, uy_txt, data_txt = farfield.load_data(txt_file_path)
        ux_mat, uy_mat, data_mat = farfield.load_data(mat_file_path)
        
        self.assertTrue(np.array_equal(ux_txt, ux_mat)) # type: ignore
        self.assertTrue(np.array_equal(uy_txt, uy_mat)) # type: ignore
        self.assertTrue((data_txt-data_mat<1e-20).all())

    def test_h5_float_dataset_loading(self):
        h5_file_path = Path('./tests/test_data/farfield_exported_test.h5')
        ref_file_path = Path('./tests/test_data/farfield_float_array.dat')

        _, _, data_h5 = farfield.load_data(h5_file_path)
        data_ref = np.loadtxt(ref_file_path)

        self.assertTrue(np.array_equal(data_h5, data_ref))


class TestApertureFunction(unittest.TestCase):
    """
    Test suite for the aperture function.
    """

    def setUp(self):
        """
        Set up mock data for the tests. This runs before each test.
        """
        self.x_data = np.linspace(0, 10, 100)
        self.y_data = 10 * np.exp(-((self.x_data - 5)**2) / (2 * 1.5**2))
        self.mock_A = 10.0
        self.mock_x0 = 5.0
        self.mock_sigma = 1.5

    @patch('emutils.farfield.fit_gaussian')
    def test_default_threshold_esqr(self, mock_fit):
        """
        Test the function with the default threshold 'esqr'.
        """
        mock_fit.return_value = (self.mock_A, self.mock_x0, self.mock_sigma)
        
        result = farfield.aperture(self.x_data, self.y_data) # Default is 'esqr'
        
        # For 'esqr', delta_x should be 2 * sigma
        expected_delta_x = 2 * self.mock_sigma
        expected_x_left = self.mock_x0 - expected_delta_x
        expected_x_right = self.mock_x0 + expected_delta_x
        
        self.assertAlmostEqual(result['threshold_points'][0], expected_x_left)
        self.assertAlmostEqual(result['threshold_points'][1], expected_x_right)
        self.assertEqual(result['fit_params']['sigma'], self.mock_sigma)

    @patch('emutils.farfield.fit_gaussian')
    def test_e_threshold(self, mock_fit):
        """
        Test the function with the 'e' threshold.
        """
        mock_fit.return_value = (self.mock_A, self.mock_x0, self.mock_sigma)
        
        result = farfield.aperture(self.x_data, self.y_data, threshold='e')
        
        # For 'e', delta_x should be sqrt(2) * sigma
        expected_delta_x = np.sqrt(2) * self.mock_sigma
        expected_x_left = self.mock_x0 - expected_delta_x
        expected_x_right = self.mock_x0 + expected_delta_x
        
        self.assertAlmostEqual(result['threshold_points'][0], expected_x_left)
        self.assertAlmostEqual(result['threshold_points'][1], expected_x_right)

    @patch('emutils.farfield.fit_gaussian')
    def test_float_threshold(self, mock_fit):
        """
        Test the function with a valid float threshold (half max).
        """
        mock_fit.return_value = (self.mock_A, self.mock_x0, self.mock_sigma)
        
        # Using H = 5 (half of A = 10)
        float_threshold = 1 / np.e
        result = farfield.aperture(self.x_data, self.y_data, threshold=float_threshold)
        
        # For a float, delta_x = sigma * sqrt(2 * ln(A/H))
        expected_delta_x = self.mock_sigma * np.sqrt(2)
        expected_x_left = self.mock_x0 - expected_delta_x
        expected_x_right = self.mock_x0 + expected_delta_x
        
        self.assertAlmostEqual(result['threshold_points'][0], expected_x_left)
        self.assertAlmostEqual(result['threshold_points'][1], expected_x_right)

    @patch('emutils.farfield.fit_gaussian')
    def test_fwhm_calculation(self, mock_fit):
        """
        Test that the FWHM is calculated correctly, regardless of threshold.
        """
        mock_fit.return_value = (self.mock_A, self.mock_x0, self.mock_sigma)
        
        result = farfield.aperture(self.x_data, self.y_data)
        
        expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * self.mock_sigma
        self.assertAlmostEqual(result['FWHM'], expected_fwhm)

    @patch('emutils.farfield.fit_gaussian')
    def test_invalid_string_threshold(self, mock_fit):
        """
        Test that an invalid string threshold raises a ValueError.
        """
        mock_fit.return_value = (self.mock_A, self.mock_x0, self.mock_sigma)
        
        with self.assertRaises(ValueError):
            farfield.aperture(self.x_data, self.y_data, threshold='invalid_string')

    @patch('emutils.farfield.fit_gaussian')
    def test_invalid_float_threshold_zero(self, mock_fit):
        """
        Test that a float threshold of 0 raises a ValueError.
        """
        mock_fit.return_value = (self.mock_A, self.mock_x0, self.mock_sigma)
        
        with self.assertRaises(ValueError):
            farfield.aperture(self.x_data, self.y_data, threshold=0.0)

    @patch('emutils.farfield.fit_gaussian')
    def test_invalid_float_threshold_above_amplitude(self, mock_fit):
        """
        Test that a float threshold greater than the amplitude raises a ValueError.
        """
        mock_fit.return_value = (self.mock_A, self.mock_x0, self.mock_sigma)
        
        with self.assertRaises(ValueError):
            farfield.aperture(self.x_data, self.y_data, threshold=11.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'])
