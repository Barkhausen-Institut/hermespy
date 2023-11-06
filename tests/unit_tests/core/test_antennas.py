# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi, speed_of_light

from hermespy.core import AntennaArray, Dipole, Direction, LinearAntenna, IdealAntenna, PatchAntenna, Transformation, UniformArray
from .test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
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

        self.assertCountEqual((2 ** -.5, 2 ** -.5), self.antenna.local_characteristics(0., 0.))


    def test_plot_polarization(self) -> None:
        """Calling the plot routine should return a figure object"""

        self.assertIsInstance(self.antenna.plot_polarization(), plt.Figure)

    def test_plot_gain(self) -> None:
        """Calling the plot routine should return a figure object"""

        self.assertIsInstance(self.antenna.plot_gain(), plt.Figure)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.antenna)
        

class TestLinearAntenna(TestCase):
        
        def setUp(self) -> None:
            
            self.antenna = LinearAntenna(slant=0.)
            
        def test_local_polarization(self) -> None:
            """Polarization should return the correct polarization for the given slant angle"""
            
            self.antenna.slant = 0.
            expected_vertical_polarization = (1., 0.)
            
            assert_array_almost_equal(expected_vertical_polarization, self.antenna.local_characteristics(0., 0.))
            assert_array_almost_equal(expected_vertical_polarization, self.antenna.local_characteristics(1., 1.))
            
            self.antenna.slant = .5 * pi
            expected_horizontal_polarization = (0., 1.)
            
            assert_array_almost_equal(expected_horizontal_polarization, self.antenna.local_characteristics(0., 0.))
            assert_array_almost_equal(expected_horizontal_polarization, self.antenna.local_characteristics(1., 1.))

        def test_global_polarization(self) -> None:
            """Global polarization should be correctly computed for different poses"""

            self.antenna.orientation = np.array([.5 * pi, 0., 0.])
            assert_array_almost_equal(np.array([0., 1.]), self.antenna.global_characteristics(Direction.From_Cartesian(np.array([1, 0, 0]))))
            assert_array_almost_equal(np.array([1., 0.]), self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 1, 0]))))
            assert_array_almost_equal(np.array([0., 1.]), self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 0, 1]))))

            self.antenna.orientation = np.array([0., 0., 0.])
            assert_array_almost_equal(np.array([1., 0.]), self.antenna.global_characteristics(Direction.From_Cartesian(np.array([1, 0, 0]))))
            assert_array_almost_equal(np.array([1., 0.]), self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 1, 0]))))
            assert_array_almost_equal(np.array([1., 0.]), self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 0, 1]))))



        def test_serialization(self) -> None:
            """Test YAML serialization"""

            test_yaml_roundtrip_serialization(self, self.antenna)

        
class TestPatchAntenna(TestCase):
    """Test the patch antenna model"""
    
    def setUp(self) -> None:
        
        self.antenna = PatchAntenna()

    def test_polarization(self) -> None:
        """Polarization should return vertical polarization"""

        self.assertCountEqual((1., 0.), self.antenna.local_characteristics(0., 0.))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.antenna)


class TestDipoleAntenna(TestCase):
    """Test the dipole antenna model"""
    
    def setUp(self) -> None:
        
        self.antenna = Dipole()

    def test_polarization(self) -> None:
        """Polarization should return vertical polarization"""

        self.assertCountEqual((0., 0.), self.antenna.local_characteristics(0., 0.))

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

    def test_init_type(self) -> None:
        """Test initializaion from antenna type"""
        
        array = UniformArray(type(self.antenna), self.spacing, self.dimensions)
        assert_array_almost_equal(self.array.topology, array.topology)

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

    def test_polarization_from_angles(self) -> None:
        """Polarization from angles should generate the correct polarization matrix"""

        local_polarization = self.array.characteristics(Direction.From_Spherical(0., 0.), 'local')
        global_polarization = self.array.characteristics(Direction.From_Spherical(0., 0.), 'global')
        
        self.assertSequenceEqual((self.array.num_antennas, 2), local_polarization.shape)
        self.assertSequenceEqual((self.array.num_antennas, 2), global_polarization.shape)

    def test_polarization_from_direction(self) -> None:
        """Polarization from direction should generate the correct polarization matrix"""

        direction = np.array([1, 0, 0], dtype=float)
        local_polarization = self.array.characteristics(direction, 'local')
        global_polarization = self.array.characteristics(direction, 'global')
        
        self.assertSequenceEqual((self.array.num_antennas, 2), local_polarization.shape)
        self.assertSequenceEqual((self.array.num_antennas, 2), global_polarization.shape)

    def test_plot_topology(self) -> None:
        """Calling the plot routine should return a figure object"""

        with patch("matplotlib.pyplot.figure") as figure_mock:
            _ = self.array.plot_topology()
            figure_mock.assert_called_once()

    def test_cartesian_phase_response_local(self) -> None:
        """Local cartesian response phase function should generate a proper sensor array response vector"""

        front_target_position = np.array([100, 0, 0])
        back_target_position = -front_target_position

        front_array_response = self.array.cartesian_phase_response(self.carrier_frequency, front_target_position, "local")
        back_array_response = self.array.cartesian_phase_response(self.carrier_frequency, back_target_position, "local")

        assert_array_almost_equal(front_array_response, back_array_response)
    
    def test_cartesian_phase_response_global(self) -> None:
        """Global cartesian response phase function should generate a proper sensor array response vector"""
        
        front_target_position = np.array([100, 0, 0])
        back_target_position = -front_target_position

        front_array_response = self.array.cartesian_phase_response(self.carrier_frequency, front_target_position, "global")
        back_array_response = self.array.cartesian_phase_response(self.carrier_frequency, back_target_position, "global")

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_cartesian_phase_response_validation(self) -> None:
        """Cartesian phase response function should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.array.cartesian_phase_response(self.carrier_frequency, np.array([1, 2, 3, 4]))

    def test_cartesian_array_response(self) -> None:
        """Cartesian array response should generate a proper sensor array response matrix"""
        
        front_target_position = np.array([100, 0, 0], dtype=float)
        back_target_position = -front_target_position

        front_array_response = self.array.cartesian_array_response(self.carrier_frequency, front_target_position)
        back_array_response = self.array.cartesian_array_response(self.carrier_frequency, back_target_position)

        assert_array_almost_equal(front_array_response, back_array_response)
        
    def test_cartesian_array_response_validation(self) -> None:
        """Cartesian array response should raise a ValueError on invalid position arguments"""
        
        with self.assertRaises(ValueError):
            _ = self.array.cartesian_array_response(self.carrier_frequency, np.arange(5))
        
    def test_horizontal_response(self) -> None:
        """Horizontal response function should generate a proper sensor array response vector"""

        elevation = 0
        azimuth = .25 * pi

        front_array_response = self.array.horizontal_phase_response(self.carrier_frequency, azimuth, elevation)
        back_array_response = self.array.horizontal_phase_response(self.carrier_frequency, azimuth - pi, elevation - pi)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_spherical_response(self) -> None:
        """Spherical response function should generate a proper sensor array response vector"""

        zenith = 0
        azimuth = .25 * pi

        front_array_response = self.array.spherical_phase_response(self.carrier_frequency, azimuth, zenith)
        back_array_response = self.array.spherical_phase_response(self.carrier_frequency, azimuth - pi, zenith - pi)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.array)


class TestAntennaArray(TestCase):
    """Test the customizable antenna array model"""
    
    def setUp(self) -> None:

        self.antennas = [
            IdealAntenna(pose=Transformation.From_RPY(np.array([0., 0., 0.]), np.array([0., 0., 0.]))),
            IdealAntenna(pose=Transformation.From_RPY(np.array([0., 0., 0.]), np.array([1., 1., 1.]))),
        ]

        self.array = AntennaArray(self.antennas)
        
    def test_antennas(self) -> None:
        """Antennas property should return the correct antennas"""
        
        self.assertCountEqual(self.antennas, self.array.antennas)
        
    def test_num_antennas(self) -> None:
        """Number of antennas property should report the correct number of antennas"""
        
        self.assertEqual(2, self.array.num_antennas)
        
    def test_num_transmit_antennas(self) -> None:
        """Number of transmit antennas property should report the correct antenna count"""
        
        self.assertEqual(2, self.array.num_transmit_antennas)
        
    def test_num_receive_antennas(self) -> None:
        """Number of receive antennas property should report the correct antenna count"""
        
        self.assertEqual(2, self.array.num_receive_antennas)
        
    def test_add_existing_antenna(self) -> None:
        """Adding an already existing antenn should do nothing"""
        
        self.array.add_antenna(self.array.antennas[0])
        self.assertEqual(2, self.array.num_antennas)
            
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
        
        polarization = self.array.characteristics(Direction.From_Spherical(0., 0.))
        self.assertCountEqual((2, 2), polarization.shape)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.array)
