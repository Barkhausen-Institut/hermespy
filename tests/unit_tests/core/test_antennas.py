# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi, speed_of_light

from hermespy.core import Antenna, AntennaArray, AntennaMode, AntennaPort, CustomAntennaArray, Dipole, Direction, LinearAntenna, IdealAntenna, PatchAntenna, UniformArray
from .test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestAntenna(TestCase):
    """Test the antenna model"""

    antenna: Antenna
    port: AntennaPort[Antenna, AntennaArray]

    def setUp(self) -> None:
        self.port = AntennaPort()
        self.antenna.port = self.port

    def test_mode_setget(self) -> None:
        """Mode property getter should return setter argument"""

        with patch.object(self.port, "antennas_updated") as antennas_updated_mock:
            mode = Mock(spec=AntennaMode)
            self.antenna.mode = mode

            self.assertEqual(mode, self.antenna.mode)
            antennas_updated_mock.assert_called_once()

    def test_port_setget(self) -> None:
        """Port property getter should return setter argument"""

        port = Mock(spec=AntennaPort)
        self.antenna.port = port

        self.assertIs(port, self.antenna.port)
        port.add_antenna.assert_called_once_with(self.antenna)
        self.assertFalse(self.antenna.is_base)

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

        test_roundtrip_serialization(self, self.antenna, {'port'})


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


class TestAntennaPort(TestCase):
    """Test the antenna port class"""

    def setUp(self) -> None:
        self.antennas = [IdealAntenna(AntennaMode.DUPLEX), IdealAntenna(AntennaMode.TX), IdealAntenna(AntennaMode.RX)]
        self.port = AntennaPort(self.antennas)

    def test_antennas(self) -> None:
        """Antenna property should return the correct sequence of antennas"""

        self.assertSequenceEqual(self.antennas, self.port.antennas)

    def test_num_antennas(self) -> None:
        """Number of antennas property should report the correct antenna count"""

        self.assertEqual(len(self.antennas), self.port.num_antennas)

        self.port.add_antenna(IdealAntenna())
        self.assertEqual(1 + len(self.antennas), self.port.num_antennas)

    def test_antennas_updated_validation(self) -> None:
        """Antennas updated callback should raise a RuntimeError on invalid antenna modes"""

        mock_antenna = IdealAntenna()
        mock_antenna.mode = "xxxx"

        with self.assertRaises(RuntimeError):
            self.port.add_antenna(mock_antenna)

    def test_add_antenna_validation(self) -> None:
        """Adding a new antenna should raise a ValueError if already assigned a port"""

        antenna = IdealAntenna()
        antenna.port = Mock()

        with self.assertRaises(ValueError):
            self.port.add_antenna(antenna)

    def test_add_antenna(self) -> None:
        """Adding a new annteann should update the antenna's port reference"""

        antenna = IdealAntenna()
        self.port.add_antenna(antenna)

        self.assertIn(antenna, self.port.antennas)
        self.assertIs(self.port, antenna.port)

    def test_remove_antenna(self) -> None:
        """Removing an antenna should update the antenna's port reference"""

        antenna = self.port.antennas[0]
        self.port.remove_antenna(antenna)

        self.assertNotIn(antenna, self.port.antennas)
        self.assertIsNone(antenna.port)

    def test_num_transmit_antennas(self) -> None:
        """Number of transmitting antennas property should return the correct value"""

        self.assertEqual(2, self.port.num_transmit_antennas)

    def test_num_receive_antennas(self) -> None:
        """Number of receiving antennas property should return the correct value"""

        self.assertEqual(2, self.port.num_receive_antennas)

    def test_transmitting(self) -> None:
        """Transmitting property should return the correct value"""

        self.assertTrue(self.port.transmitting)

        for antenna in self.port.antennas:
            antenna.mode = AntennaMode.RX

        self.assertFalse(self.port.transmitting)

    def test_receiving(self) -> None:
        """Receiving property should return the correct value"""

        self.assertTrue(self.port.receiving)

        for antenna in self.port.antennas:
            antenna.mode = AntennaMode.TX

        self.assertFalse(self.port.receiving)

    def test_transmit_antennas(self) -> None:
        """Transmit antennas property should return the correct sequence of antennas"""

        self.assertSequenceEqual(self.antennas[:2], self.port.transmit_antennas)

    def test_receive_antennas(self) -> None:
        """Receive antennas property should return the correct sequence of antennas"""

        self.assertSequenceEqual(self.antennas[::2], self.port.receive_antennas)

    def test_array(self) -> None:
        """Array property should return the correct array"""

        array = Mock()

        self.port.array = array
        self.assertIs(array, self.port.array)

        self.port.array = array
        self.assertIs(array, self.port.array)

    def test_serialization(self) -> None:
        """Test antenna port serialization"""

        test_roundtrip_serialization(self, self.port)


class _TestAntennaArray(TestCase):
    """Test the base class for all antenna array models"""

    array: AntennaArray[AntennaPort[Antenna, AntennaArray], Antenna]

    def test_num_ports(self) -> None:
        """Number of ports property should return the correct value"""

        expected_num_ports = len(self.array.ports)
        self.assertEqual(expected_num_ports, self.array.num_ports)

    def test_num_transmit_ports(self) -> None:
        """Number of transmitting ports property should return the correct value"""

        expected_num_ports = len(self.array.transmit_ports)
        self.assertEqual(expected_num_ports, self.array.num_transmit_ports)

    def test_num_receive_ports(self) -> None:
        """Number of receiving ports property should return the correct value"""

        expected_num_ports = len(self.array.receive_ports)
        self.assertEqual(expected_num_ports, self.array.num_receive_ports)

    def test_transmit_ports(self) -> None:
        """Transmit ports property should return only transmitting ports"""

        ports = self.array.transmit_ports
        for port in ports:
            transmitting = False
            for antenna in port.antennas:
                if antenna.mode == AntennaMode.TX or antenna.mode == AntennaMode.DUPLEX:
                    transmitting = True
                    break
            self.assertTrue(transmitting)

    def test_receive_ports(self) -> None:
        """Receive ports property should return only receiving ports"""

        ports = self.array.receive_ports
        for port in ports:
            receiving = False
            for antenna in port.antennas:
                if antenna.mode == AntennaMode.RX or antenna.mode == AntennaMode.DUPLEX:
                    receiving = True
                    break
            self.assertTrue(receiving)

    def test_num_antennas(self) -> None:
        """Number of antennas property should return the correct value"""

        expected_num_antennas = sum([port.num_antennas for port in self.array.ports])
        self.assertEqual(expected_num_antennas, self.array.num_antennas)

    def test_num_transmit_antennas(self) -> None:
        """Number of transmit antennas property should return the correct value"""

        expected_num_antennas = sum([port.num_transmit_antennas for port in self.array.ports])
        self.assertEqual(expected_num_antennas, self.array.num_transmit_antennas)

    def test_num_receive_antennas(self) -> None:
        """Number of receive antennas property should return the correct value"""

        expected_num_antennas = sum([port.num_receive_antennas for port in self.array.ports])
        self.assertEqual(expected_num_antennas, self.array.num_receive_antennas)

    def test_count_antennas(self) -> None:
        """Antenna counting function should return the correct value"""

        selected_ports = [self.array.num_ports - 1,]
        expected_num_antennas = self.array.ports[-1].num_antennas

        num_antennas = self.array.count_antennas(selected_ports)
        self.assertEqual(expected_num_antennas, num_antennas)

    def test_count_transmit_antennas(self) -> None:
        """Transmit antenna counting function should return the correct value"""

        selected_ports = [self.array.num_transmit_ports - 1,]
        expected_num_antennas = self.array.transmit_ports[-1].num_transmit_antennas

        num_antennas = self.array.count_transmit_antennas(selected_ports)
        self.assertEqual(expected_num_antennas, num_antennas)

    def test_count_receive_antennas(self) -> None:
        """Receive antenna counting function should return the correct value"""

        selected_ports = [self.array.num_receive_ports - 1,]
        expected_num_antennas = self.array.receive_ports[-1].num_receive_antennas

        num_antennas = self.array.count_receive_antennas(selected_ports)
        self.assertEqual(expected_num_antennas, num_antennas)

    def test_antennas(self) -> None:
        """Antennas property should return the correct sequence of antennas"""

        expected_antennas = []
        for port in self.array.ports:
            expected_antennas.extend(port.antennas)

        self.assertSequenceEqual(expected_antennas, self.array.antennas)

    def test_transmit_antennas(self) -> None:
        """Transmit antennas property should return the correct sequence of antennas"""

        expected_antennas = []
        for port in self.array.transmit_ports:
            expected_antennas.extend(port.antennas)

        self.assertSequenceEqual(expected_antennas, self.array.transmit_antennas)

    def test_receive_antennas(self) -> None:
        """Receive antennas property should return the correct sequence of antennas"""

        expected_antennas = []
        for port in self.array.receive_ports:
            expected_antennas.extend(port.receive_antennas)

        self.assertSequenceEqual(expected_antennas, self.array.receive_antennas)

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

    def test_serialization(self) -> None:
        """Test antenna array serialization"""

        test_roundtrip_serialization(self, self.array)


class TestUniformArray(_TestAntennaArray):
    """Test the Uniform array model"""

    array: UniformArray[AntennaPort[Antenna, AntennaArray], Antenna]

    def setUp(self) -> None:
        self.antenna = IdealAntenna()
        self.carrier_frequency = 1e-9
        self.wavelength = self.carrier_frequency / speed_of_light
        self.spacing = 0.5 * self.wavelength
        self.dimensions = (10, 9, 8)

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

    def test_num_antennas(self) -> None:
        """The number of antennas property should report the correct antenna count"""

        super().test_num_antennas()
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

    array: CustomAntennaArray[AntennaPort[Antenna, AntennaArray], Antenna]

    def setUp(self) -> None:
        self.ports = [AntennaPort([IdealAntenna(AntennaMode.DUPLEX), IdealAntenna(AntennaMode.TX)]), IdealAntenna(AntennaMode.RX)]
        self.array = CustomAntennaArray(self.ports)

    def test_init(self) -> None:
        """Initialization routine should properly assign the ports"""

        self.assertEqual(2, self.array.num_ports)
        self.assertEqual(3, self.array.num_antennas)

    def test_ports(self) -> None:
        """Ports property should return the correct sequence of ports"""

        ports = self.array.ports
        self.assertIs(self.ports[0], ports[0])
        self.assertIs(self.ports[1], ports[1].antennas[0])

    def test_add_port(self) -> None:
        """Adding a new port should correctly adapt the array"""

        port = AntennaPort([IdealAntenna(AntennaMode.DUPLEX)])
        self.array.add_port(port)

        self.assertEqual(3, self.array.num_ports)
        self.assertEqual(4, self.array.num_antennas)
        self.assertIs(self.array, port.array)

        # Re-adding the port should do nothing
        self.array.add_port(port)

        self.assertEqual(3, self.array.num_ports)
        self.assertEqual(4, self.array.num_antennas)
        self.assertIs(self.array, port.array)

    def test_remove_port_validation(self) -> None:
        """Removing a port should raise a ValueError if not assigned to the array"""

        port = AntennaPort([IdealAntenna(AntennaMode.DUPLEX)])

        with self.assertRaises(ValueError):
            self.array.remove_port(port)

    def test_remove_port(self) -> None:
        """Removing a port should correctly adapt the array"""

        port = self.ports[0]
        self.array.remove_port(port)

        self.assertEqual(1, self.array.num_ports)
        self.assertEqual(1, self.array.num_antennas)
        self.assertIsNone(port.array)

    def test_add_antenna_validation(self) -> None:
        """Adding an antenna should raise a ValueError if already assigned to the array"""

        antenna = self.ports[0].antennas[0]

        with self.assertRaises(ValueError):
            self.array.add_antenna(antenna)

    def test_add_antenna(self) -> None:
        """Adding an antenna should correctly adapt the array"""

        antenna = IdealAntenna()
        added_port = self.array.add_antenna(antenna)

        self.assertEqual(3, self.array.num_ports)
        self.assertEqual(4, self.array.num_antennas)
        self.assertIn(antenna, self.array.antennas)
        self.assertIn(added_port, self.array.ports)


del _TestAntenna
del _TestAntennaArray
