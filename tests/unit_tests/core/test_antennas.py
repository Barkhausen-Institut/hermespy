# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi, speed_of_light

from hermespy.core import Antenna, AntennaArray, AntennaMode, CustomAntennaArray, Dipole, Direction, LinearAntenna, IdealAntenna, PatchAntenna, Transformation, UniformArray
from .test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestAntenna(TestCase):
    """Test the antenna model"""

    antenna: Antenna

    def test_mode_setget(self) -> None:
        """Mode property getter should return setter argument"""

        mode = Mock(spec=AntennaMode)
        self.antenna.mode = mode
        self.assertEqual(mode, self.antenna.mode)

    def test_global_characteristics_dimensions(self) -> None:
        """Global characteristics should return the correct polarization"""

        self.antenna.orientation = np.array([0.0, 0.0, 0.0])
        self.antenna.position = np.array([0.0, 0.0, 0.0])

        unit_transformation_x_characteristics = self.antenna.global_characteristics(Direction.From_Cartesian(np.array([1, 0, 0])))
        unit_transformation_y_characteristics = self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 1, 0])))
        unit_transformation_z_characteristics = self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 0, 1])))

        self.antenna.orientation = np.array([0.0, 0.0, 0.5 * pi])
        x_rotation_x_characteristics = self.antenna.global_characteristics(Direction.From_Cartesian(np.array([1, 0, 0])))
        x_rotation_y_characteristics = self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 1, 0])))
        x_rotation_z_characteristics = self.antenna.global_characteristics(Direction.From_Cartesian(np.array([0, 0, 1])))

        self.assertSequenceEqual((2,), unit_transformation_x_characteristics.shape)
        self.assertSequenceEqual((2,), unit_transformation_y_characteristics.shape)
        self.assertSequenceEqual((2,), unit_transformation_z_characteristics.shape)
        self.assertSequenceEqual((2,), x_rotation_x_characteristics.shape)
        self.assertSequenceEqual((2,), x_rotation_y_characteristics.shape)
        self.assertSequenceEqual((2,), x_rotation_z_characteristics.shape)

    def test_plot_polarization(self) -> None:
        """Calling the ploarization plot routine should return a figure object"""

        with patch("matplotlib.pyplot.figure") as figure_mock:
            _ = self.antenna.plot_polarization()
            figure_mock.assert_called_once()

    def test_plot_gain(self) -> None:
        """Calling the gain plot routine should return a figure object"""

        with patch("matplotlib.pyplot.figure") as figure_mock:
            _ = self.antenna.plot_gain()
            figure_mock.assert_called_once()

    def test_serialization(self) -> None:
        """Test antenna serialization"""

        test_roundtrip_serialization(self, self.antenna)


class TestIdealAntenna(_TestAntenna):
    """Test the ideal antenna model"""

    def setUp(self) -> None:
        self.antenna = IdealAntenna()
        super().setUp()

    def test_local_characteristics(self) -> None:
        """Polarization should return unit polarization"""

        self.assertCountEqual((2**-0.5, 2**-0.5), self.antenna.local_characteristics(0.0, 0.0))


class TestLinearAntenna(TestCase):
    def setUp(self) -> None:
        self.antenna = LinearAntenna(slant=0.0)
        super().setUp()

    def test_local_polarization(self) -> None:
        """Polarization should return the correct polarization for the given slant angle"""

        self.antenna.slant = 0.0
        expected_vertical_polarization = (1.0, 0.0)

        assert_array_almost_equal(expected_vertical_polarization, self.antenna.local_characteristics(0.0, 0.0))
        assert_array_almost_equal(expected_vertical_polarization, self.antenna.local_characteristics(1.0, 1.0))

        self.antenna.slant = 0.5 * pi
        expected_horizontal_polarization = (0.0, 1.0)

        assert_array_almost_equal(expected_horizontal_polarization, self.antenna.local_characteristics(0.0, 0.0))
        assert_array_almost_equal(expected_horizontal_polarization, self.antenna.local_characteristics(1.0, 1.0))

    def test_copy_antenna(self) -> None:
        """Test copying the antenna"""

        copy = self.antenna.copy()
        self.assertEqual(self.antenna.mode, copy.mode)
        self.assertEqual(self.antenna.slant, copy.slant)
        assert_array_equal(self.antenna.pose, copy.pose)


class TestPatchAntenna(_TestAntenna):
    """Test the patch antenna model"""

    def setUp(self) -> None:
        self.antenna = PatchAntenna()
        super().setUp()

    def test_local_characteristics(self) -> None:
        """Polarization should return vertical polarization"""

        self.assertCountEqual((1.0, 0.0), self.antenna.local_characteristics(0.0, 0.0))

    def test_copy_antenna(self) -> None:
        """Test copying the antenna"""

        copy = self.antenna.copy()
        self.assertEqual(self.antenna.mode, copy.mode)
        assert_array_equal(self.antenna.pose, copy.pose)


class TestDipoleAntenna(_TestAntenna):
    """Test the dipole antenna model"""

    def setUp(self) -> None:
        self.antenna = Dipole()
        super().setUp()

    def test_local_characteristics(self) -> None:
        """Polarization should return vertical polarization"""

        self.assertCountEqual((0.0, 0.0), self.antenna.local_characteristics(0.0, 0.0))

    def test_copy_antenna(self) -> None:
        """Test copying the antenna"""

        copy = self.antenna.copy()
        self.assertEqual(self.antenna.mode, copy.mode)
        assert_array_equal(self.antenna.pose, copy.pose)



class _TestAntennaArray(TestCase):
    """Test the base class for all antenna array models"""

    array: AntennaArray[Antenna]
    expected_num_transmit_antennas: int
    expected_num_receive_antennas: int
    expected_num_antennas: int

    def test_num_antennas(self) -> None:
        """Number of antennas property should return the correct value"""

        self.assertEqual(self.expected_num_antennas, self.array.num_antennas)

    def test_num_transmit_antennas(self) -> None:
        """Number of transmit antennas property should return the correct value"""

        self.assertEqual(self.expected_num_transmit_antennas, self.array.num_transmit_antennas)

    def test_num_receive_antennas(self) -> None:
        """Number of receive antennas property should return the correct value"""

        self.assertEqual(self.expected_num_receive_antennas, self.array.num_receive_antennas)

    def test_antennas(self) -> None:
        """Antennas property should return the correct sequence of antennas"""

        self.assertEqual(self.expected_num_antennas, len(self.array.antennas))

    def test_transmit_antennas(self) -> None:
        """Transmit antennas property should return the correct sequence of antennas"""

        self.assertEqual(self.expected_num_transmit_antennas, len(self.array.transmit_antennas))

    def test_receive_antennas(self) -> None:
        """Receive antennas property should return the correct sequence of antennas"""

        self.assertEqual(self.expected_num_receive_antennas, len(self.array.receive_antennas))

    def test_topology_empty_dimensions(self) -> None:
        """Topology property should return the correct dimensions for an empty array"""

        empty_array = CustomAntennaArray()
        self.assertSequenceEqual((0, 3), empty_array.topology.shape)

    def test_topology_dimensions(self) -> None:
        """Topology property should return the correct dimensions"""

        self.assertSequenceEqual((self.array.num_antennas, 3), self.array.topology.shape)

    def test_transmit_topology_empty(self) -> None:
        """Transmit topology property should return the correct dimensions for an empty array"""

        empty_array = CustomAntennaArray()
        self.assertSequenceEqual((0, 3), empty_array.transmit_topology.shape)

    def test_transmit_topology_dimensions(self) -> None:
        """Transmit topology property should return the correct dimensions"""

        self.assertSequenceEqual((self.array.num_transmit_antennas, 3), self.array.transmit_topology.shape)

    def test_receive_topology_empty(self) -> None:
        """Receive topology property should return the correct dimensions for an empty array"""

        empty_array = CustomAntennaArray()
        self.assertSequenceEqual((0, 3), empty_array.receive_topology.shape)

    def test_receive_topology_dimensions(self) -> None:
        """Receive topology property should return the correct dimensions"""

        self.assertSequenceEqual((self.array.num_receive_antennas, 3), self.array.receive_topology.shape)

    def test_topology_from_mode_validation(self) -> None:
        """Topology from mode function should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.array._topology("wrongmode")

    def test_topology_transformation_invariance(self) -> None:
        """Topology property should not change if array transformation changes"""

        initial_topology = self.array.topology

        self.array.position = np.array([5.0, -2.0, 7.0])
        self.array.orientation = np.array([0.5, -0.5, 1])
        translated_topology = self.array.topology

        assert_array_almost_equal(initial_topology, translated_topology)

    def test_transmit_topology_transformation_invariance(self) -> None:
        """Transmit topology property should not change if array transformation changes"""

        initial_topology = self.array.transmit_topology

        self.array.position = np.array([5.0, -2.0, 7.0])
        self.array.orientation = np.array([0.5, -0.5, 1])
        translated_topology = self.array.transmit_topology

        assert_array_almost_equal(initial_topology, translated_topology)

    def test_receive_topology_transformation_invariance(self) -> None:
        """Receive topology property should not change if array position changes"""

        initial_topology = self.array.receive_topology

        self.array.position = np.array([5.0, -2.0, 7.0])
        self.array.orientation = np.array([0.5, -0.5, 1])
        translated_topology = self.array.receive_topology

        assert_array_almost_equal(initial_topology, translated_topology)

    def test_characteristics_validation(self) -> None:
        """Characteristics function should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.array.characteristics(np.array([1, 2, 3, 4]), "wrongmode")

        with self.assertRaises(ValueError):
            _ = self.array.characteristics(np.array([1, 2, 3]), "wrongmode")

    def test_characteristics_dimensions(self) -> None:
        """Characteristics function should return an array of correct dimensionality"""

        location = np.array([1.0, 2.0, 3.0])
        direction = Direction.From_Cartesian(location, normalize=True)
        for arg_0, frame in zip((location, direction), ("local", "global")):
            duplex_characteristics = self.array.characteristics(arg_0, AntennaMode.DUPLEX, frame)
            tx_characteristics = self.array.characteristics(arg_0, AntennaMode.TX, frame)
            rx_characteristics = self.array.characteristics(arg_0, AntennaMode.RX, frame)

            self.assertSequenceEqual((self.array.num_antennas, 2), duplex_characteristics.shape)
            self.assertSequenceEqual((self.array.num_transmit_antennas, 2), tx_characteristics.shape)
            self.assertSequenceEqual((self.array.num_receive_antennas, 2), rx_characteristics.shape)

    def test_plot_topology(self) -> None:
        """Calling the plot routine should return a figure object"""

        with patch("matplotlib.pyplot.figure") as figure_mock:
            for mode in (AntennaMode.DUPLEX, AntennaMode.TX, AntennaMode.RX):
                _ = self.array.plot_topology(mode=mode)
                figure_mock.assert_called_once()
                figure_mock.reset_mock()

    def test_cartesian_phase_response_validation(self) -> None:
        """Cartesian phase response function should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.array.cartesian_phase_response(1.0, np.array([1, 2, 3, 4]))

    def test_cartesian_phase_response_dimensions(self) -> None:
        """Cartesian phase response function should return an array of correct dimensionality"""

        position = np.array([1.0, 2.0, 3.0])
        for frame in ("local", "global"):
            duplex_phase_response = self.array.cartesian_phase_response(1.0, position, frame, AntennaMode.DUPLEX)
            tx_phase_response = self.array.cartesian_phase_response(1.0, position, frame, AntennaMode.TX)
            rx_phase_response = self.array.cartesian_phase_response(1.0, position, frame, AntennaMode.RX)

            self.assertSequenceEqual((self.array.num_antennas,), duplex_phase_response.shape)
            self.assertSequenceEqual((self.array.num_transmit_antennas,), tx_phase_response.shape)
            self.assertSequenceEqual((self.array.num_receive_antennas,), rx_phase_response.shape)

    def test_cartesian_phase_response_local(self) -> None:
        """Local cartesian response phase function should generate a proper sensor array response vector"""

        front_target_position = np.array([100, 0, 0])
        back_target_position = -front_target_position

        front_array_response = self.array.cartesian_phase_response(1.234, front_target_position, "local")
        back_array_response = self.array.cartesian_phase_response(1.234, back_target_position, "local")

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_cartesian_phase_response_global(self) -> None:
        """Global cartesian response phase function should generate a proper sensor array response vector"""

        front_target_position = np.array([100, 0, 0])
        back_target_position = -front_target_position

        front_array_response = self.array.cartesian_phase_response(1.234, front_target_position, "global")
        back_array_response = self.array.cartesian_phase_response(1.234, back_target_position, "global")

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_cartesian_array_response_validation(self) -> None:
        """Cartesian array response should raise a ValueError on invalid position arguments"""

        with self.assertRaises(ValueError):
            _ = self.array.cartesian_array_response(1.0, np.arange(5))

    def test_cartesian_array_response_dimensions(self) -> None:
        """Cartesian array response should return an array of correct dimensionality"""

        position = np.array([1.0, 2.0, 3.0])
        for frame in ("local", "global"):
            duplex_array_response = self.array.cartesian_array_response(1.0, position, frame, AntennaMode.DUPLEX)
            tx_array_response = self.array.cartesian_array_response(1.0, position, frame, AntennaMode.TX)
            rx_array_response = self.array.cartesian_array_response(1.0, position, frame, AntennaMode.RX)

            self.assertSequenceEqual((self.array.num_antennas, 2), duplex_array_response.shape)
            self.assertSequenceEqual((self.array.num_transmit_antennas, 2), tx_array_response.shape)
            self.assertSequenceEqual((self.array.num_receive_antennas, 2), rx_array_response.shape)

    def test_cartesian_array_response_rotation(self) -> None:
        """Cartesian array response should be invariant to pi rotations"""

        front_target_position = np.array([100, 0, 0], dtype=float)
        back_target_position = -front_target_position

        front_array_response = self.array.cartesian_array_response(1.234, front_target_position)
        back_array_response = self.array.cartesian_array_response(1.234, back_target_position)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_horizontal_phase_response_dimensions(self) -> None:
        """Horizontal phase response function should return an array of correct dimensionality"""

        duplex_phase_response = self.array.horizontal_phase_response(1.0, 0.0, 0.0, AntennaMode.DUPLEX)
        tx_phase_response = self.array.horizontal_phase_response(1.0, 0.0, 0.0, AntennaMode.TX)
        rx_phase_response = self.array.horizontal_phase_response(1.0, 0.0, 0.0, AntennaMode.RX)

        self.assertSequenceEqual((self.array.num_antennas,), duplex_phase_response.shape)
        self.assertSequenceEqual((self.array.num_transmit_antennas,), tx_phase_response.shape)
        self.assertSequenceEqual((self.array.num_receive_antennas,), rx_phase_response.shape)

    def test_horizontal_phase_response_rotation(self) -> None:
        """Horizontal phase response should be invariant to pi rotations"""

        elevation = 0
        azimuth = 0.25 * pi

        front_array_response = self.array.horizontal_phase_response(1.234, azimuth, elevation)
        back_array_response = self.array.horizontal_phase_response(1.234, azimuth - pi, elevation - pi)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_spherical_phase_response_dimensions(self) -> None:
        """Spherical phase response function should return an array of correct dimensionality"""

        duplex_phase_response = self.array.spherical_phase_response(1.0, 0.0, 0.0, AntennaMode.DUPLEX)
        tx_phase_response = self.array.spherical_phase_response(1.0, 0.0, 0.0, AntennaMode.TX)
        rx_phase_response = self.array.spherical_phase_response(1.0, 0.0, 0.0, AntennaMode.RX)

        self.assertSequenceEqual((self.array.num_antennas,), duplex_phase_response.shape)
        self.assertSequenceEqual((self.array.num_transmit_antennas,), tx_phase_response.shape)
        self.assertSequenceEqual((self.array.num_receive_antennas,), rx_phase_response.shape)

    def test_spherical_phase_response_rotation(self) -> None:
        """Spherical phase response should be invariant to pi rotations"""

        zenith = 0
        azimuth = 0.25 * pi

        front_array_response = self.array.spherical_phase_response(1.234, azimuth, zenith)
        back_array_response = self.array.spherical_phase_response(1.234, azimuth - pi, zenith - pi)

        assert_array_almost_equal(front_array_response, back_array_response)

    def test_state(self) -> None:
        """State function should return the correct state"""

        state = self.array.state(Transformation.From_Translation([0.0, 0.0, 0.0]))
        self.assertEqual(self.array.num_antennas, state.num_antennas)
        assert_array_equal(self.array.position, state.position)
        assert_array_equal(self.array.orientation, state.orientation)
        assert_array_equal(self.array.topology, state.topology)

    def test_serialization(self) -> None:
        """Test antenna array serialization"""

        test_roundtrip_serialization(self, self.array)


class TestUniformArray(_TestAntennaArray):
    """Test the Uniform array model"""

    array: UniformArray[Antenna]

    def setUp(self) -> None:
        self.antenna = IdealAntenna()
        self.carrier_frequency = 1e-9
        self.wavelength = self.carrier_frequency / speed_of_light
        self.spacing = 0.5 * self.wavelength
        self.dimensions = (10, 9, 8)

        self.expected_num_transmit_antennas = int(2 * 720 / 3)
        self.expected_num_receive_antennas = int(2 * 720 / 3)
        self.expected_num_antennas = 720
        self.array = UniformArray(self.antenna, self.spacing, self.dimensions)

        # Alter antenna modes for better debugging
        for a, antenna in enumerate(self.array.antennas):
            antenna.mode = AntennaMode(a % 3)

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
            self.array.spacing = -1.0

        with self.assertRaises(ValueError):
            self.array.spacing = 0.0

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

    def test_topology(self) -> None:
        """The generated topology should be uniform"""

        dimensions = 5
        spacing = 1.0
        expected_topology = np.zeros((dimensions, 3), dtype=float)
        expected_topology[:, 0] = spacing * np.arange(dimensions)

        self.array.dimensions = dimensions
        self.array.spacing = spacing

        assert_array_equal(expected_topology, self.array.topology)


class TestCustomAntennaArray(_TestAntennaArray):
    """Test the customizable antenna array model"""

    array: CustomAntennaArray[Antenna]

    def setUp(self) -> None:
        self.elements = [IdealAntenna(AntennaMode.TX), IdealAntenna(AntennaMode.RX), IdealAntenna(AntennaMode.DUPLEX)]
        self.array = CustomAntennaArray(self.elements)
        self.expected_num_transmit_antennas = 2
        self.expected_num_receive_antennas = 2
        self.expected_num_antennas = 3

    def test_init(self) -> None:
        """Initialization routine should properly assign the ports"""

        self.assertEqual(3, self.array.num_antennas)

    def test_add_antenna(self) -> None:
        """Adding an antenna should correctly adapt the array"""

        antenna = IdealAntenna()
        added_port = self.array.add_antenna(antenna)

        self.assertEqual(4, self.array.num_antennas)
        self.assertIn(antenna, self.array.antennas)


del _TestAntenna
del _TestAntennaArray
