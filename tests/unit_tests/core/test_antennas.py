from unittest import TestCase
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi, speed_of_light

from hermespy.core import AntennaArray, Dipole, IdealAntenna, PatchAntenna, UniformArray
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestIdealAntenna(TestCase):
    """Test the ideal antenna model"""

    def setUp(self) -> None:

        self.antenna = IdealAntenna()

    def test_array_setget(self) -> None:
        """Array property getter should return setter argument"""

        array = Mock()
        self.antenna.array = array

        self.assertIs(array, self.antenna.array)

        self.antenna.array = array
        self.assertIs(array, self.antenna.array)

    def test_array_removal(self) -> None:
        """Removing an array should envoke the callback"""

        array = Mock()
        self.antenna.array = array
        self.antenna.array = None

        self.assertEqual(1, array.remove_antenna.call_count)

    def test_pos_set(self) -> None:
        """Setting an antenna element position should call the respective array callback"""

        array = Mock()
        self.antenna.array = array
        self.antenna.pos = [1, 2, 3]

        self.assertEqual(1, array.set_antenna_position.call_count)

    def test_pos_set_validation(self) -> None:
        """Setting the position of a floating antenna should raise an exception"""

        with self.assertRaises(RuntimeError):
            self.antenna.pos = np.array([1, 2, 3])

    def test_transmit(self) -> None:
        """Transmitting should just be a stub returning the argument"""

        signal = Mock()
        self.assertIs(signal, self.antenna.transmit(signal))

    def test_receive(self) -> None:
        """Receiving should just be a stub returning the argument"""

        signal = Mock()
        self.assertIs(signal, self.antenna.receive(signal))

    def test_polarization(self) -> None:
        """Polarization should return unit polarization"""

        self.assertCountEqual((2 ** -.5, 2 ** -.5), self.antenna.polarization(0., 0.))

    def test_plot_polarization(self) -> None:
        """Calling the plot routine should return a figure object"""

        self.assertIsInstance(self.antenna.plot_polarization(), plt.Figure)

    def test_plot_gain(self) -> None:
        """Calling the plot routine should return a figure object"""

        self.assertIsInstance(self.antenna.plot_gain(), plt.Figure)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.antenna)
        
        
class TestPatchAntenna(TestCase):
    """Test the patch antenna model"""
    
    def setUp(self) -> None:
        
        self.antenna = PatchAntenna()

    def test_polarization(self) -> None:
        """Polarization should return vertical polarization"""

        self.assertCountEqual((1., 0.), self.antenna.polarization(0., 0.))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.antenna)


class TestDipoleAntenna(TestCase):
    """Test the dipole antenna model"""
    
    def setUp(self) -> None:
        
        self.antenna = Dipole()

    def test_polarization(self) -> None:
        """Polarization should return vertical polarization"""

        self.assertCountEqual((0., 0.), self.antenna.polarization(0., 0.))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.antenna)


class TestUniformArray(TestCase):
    """Test the Uniform array model"""

    def setUp(self) -> None:

        self.antenna = IdealAntenna()
        self.carrier_frequency = 1e-9
        self.wavelength = self.carrier_frequency / speed_of_light
        self.spacing = .5 * self.wavelength
        self.dimensions = (10, 9, 8)

        self.array = UniformArray(self.antenna, self.spacing, self.dimensions)

    def test_spacing_setget(self) -> None:
        """Spacing property getter should return setter argument"""

        spacing = 1.234
        self.array.spacing = spacing

        self.assertEqual(spacing, self.array.spacing)

    def test_spacing_validation(self) -> None:
        """Spacing property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.array.spacing = -1.

        with self.assertRaises(ValueError):
            self.array.spacing = 0.

    def test_num_antennas(self) -> None:
        """The number of antennas property should report the correct antenna count"""

        self.assertEqual(720, self.array.num_antennas)

    def test_dimensions_setget(self) -> None:
        """The dimensions property getter should return the proper antenna count"""

        self.array.dimensions = 1
        self.assertCountEqual((1, 1, 1), self.array.dimensions)

        self.array.dimensions = (1, 2)
        self.assertCountEqual((1, 2, 1), self.array.dimensions)

        self.array.dimensions = (1, 2, 3)
        self.assertCountEqual((1, 2, 3), self.array.dimensions)

    def test_dimensions_validation(self) -> None:
        """The dimensions property setter should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.array.dimensions = (1, 2, 3, 4)

        with self.assertRaises(ValueError):
            self.array.dimensions = (1, 2, -1)

    def test_topology(self) -> None:

        dimensions = 5
        spacing = 1.
        expected_topology = np.zeros((dimensions, 3), dtype=float)
        expected_topology[:, 0] = spacing * np.arange(dimensions)

        self.array.dimensions = dimensions
        self.array.spacing = spacing

        assert_array_equal(expected_topology, self.array.topology)

    def test_polarization(self) -> None:
        """The polarization should compute the correct polarization array"""

        polarization = self.array.polarization(0., 0.)
        self.assertCountEqual((10 * 9 * 8, 2), polarization.shape)
        self.assertTrue(np.any(polarization == 2 ** -.5))

    def test_plot_topology(self) -> None:
        """Calling the plot routine should return a figure object"""

        self.assertIsInstance(self.array.plot_topology(), plt.Figure)

    def test_cartesian_response(self) -> None:
        """Cartesian response function should generate a proper sensor array response vector"""

        front_target_position = np.array([100, 0, 0])
        back_target_position = -front_target_position

        front_array_response = self.array.cartesian_response(self.carrier_frequency, front_target_position)
        back_array_response = self.array.cartesian_response(self.carrier_frequency, back_target_position)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_cartesian_response_validation(self) -> None:
        """Cartesian response function should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.array.cartesian_response(self.carrier_frequency, np.array([1, 2, 3, 4]))

    def test_horizontal_response(self) -> None:
        """Horizontal response function should generate a proper sensor array response vector"""

        elevation = 0
        azimuth = .25 * pi

        front_array_response = self.array.horizontal_response(self.carrier_frequency, azimuth, elevation)
        back_array_response = self.array.horizontal_response(self.carrier_frequency, azimuth - pi, elevation - pi)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_spherical_response(self) -> None:
        """Spherical response function should generate a proper sensor array response vector"""

        zenith = 0
        azimuth = .25 * pi

        front_array_response = self.array.spherical_response(self.carrier_frequency, azimuth, zenith)
        back_array_response = self.array.spherical_response(self.carrier_frequency, azimuth - pi, zenith - pi)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.array)


class TestAntennaArray(TestCase):
    """Test the customizable antenna array model"""
    
    def setUp(self) -> None:

        self.antennas = [IdealAntenna(), IdealAntenna()]
        self.positions = [np.array([0., 0., 0.]), np.array([1., 1., 1.])]
        self.orientations = [np.array([0., 0.]), np.array([0., 0.])]

        self.array = AntennaArray(self.antennas, self.positions, self.orientations)
        
    def test_init_validation(self) -> None:
        """Initialization should raise a ValueError on unequal argument lengths"""
        
        with self.assertRaises(ValueError):
            self.array = AntennaArray(self.antennas[1:], self.positions, self.orientations)
        
    def test_antennas(self) -> None:
        """Antennas property should return the correct antennas"""
        
        self.assertCountEqual(self.antennas, self.array.antennas)
        
    def test_num_antennas(self) -> None:
        """Number of antennas property should report the correct number of antennas"""
        
        self.assertEqual(2, self.array.num_antennas)

    def test_add_antenna_validation(self) -> None:
        """Adding a new antenna should raise exceptions for invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.array.add_antenna(Mock(), np.zeros(2), Mock())
            
        with self.assertRaises(ValueError):
            self.array.add_antenna(Mock(), np.zeros(3), np.zeros(1))
            
    def test_remove_antenna(self) -> None:
        """Removing an antenna should correctly adapt the array"""
        
        self.array.remove_antenna(self.antennas[0])
        self.assertIsNone(self.antennas[0].array)
        
        self.array.remove_antenna(self.antennas[0])
        self.assertIsNone(self.antennas[0].array)
        
    def test_topology(self) -> None:
        """The generated topology should have the correct dimensions"""
        
        self.assertCountEqual((2, 3), self.array.topology.shape)
        
    def test_polarization(self) -> None:
        """The polarization pattern should be correctly generated"""
        
        polarization = self.array.polarization(0., 0.)
        self.assertCountEqual((2, 2), polarization.shape)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.array)
