# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.beamforming import BeamFocus, BeamformerBase, CoordinateFocus, DeviceFocus, ReceiveBeamformer, SphericalFocus, TransmitBeamformer
from hermespy.core import AntennaArray, Direction, Device, FloatingError, Signal
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockBeamFocus(BeamFocus):
    """Mock class to test the beam focus base class"""

    @property
    def spherical_angles(self) -> np.ndarray:
        return np.arange(2)

    def copy(self) -> BeamFocus:
        return MockBeamFocus()


class TestBeamFocus(TestCase):
    """Test the beam focus base class"""

    def setUp(self) -> None:

        self.focus = MockBeamFocus()

    def test_str_representation(self) -> None:
        """String representation should return the correct string"""

        self.assertIsInstance(str(self.focus), str)


class _TestBeamFocus(TestCase):
    """Test classes inheriting from the beam focus base class"""

    focus: BeamFocus

    def test_copy(self) -> None:
        """Copy routine should return a copy of the beam focus"""

        copy = self.focus.copy()

        self.assertIsNot(copy, self.focus)

    def test_yaml_serialization(self) -> None:
        """Beam focus should be serializable to and from YAML"""

        test_yaml_roundtrip_serialization(self, self.focus)


class TestDeviceFocus(_TestBeamFocus):
    """Test device focusing beam focus class."""

    def setUp(self) -> None:
        self.focused_device = SimulatedDevice()
        self.focused_device.position = np.arange(3)

        self.focus = DeviceFocus(self.focused_device)

        self.beamformer = Mock(spec=BeamformerBase)
        self.beamformer.operator = Mock()
        self.beamformer.operator.device = Mock(spec=Device)
        self.beamformer.operator.device.global_position = np.arange(1, 4)
        self.focus.beamformer = self.beamformer

    def test_spherical_angles_validation(self) -> None:
        """Spherical angles property getter should raise RuntimeErrors on invalid configurations"""

        self.beamformer.operator.device = None
        with self.assertRaises(RuntimeError):
            _ = self.focus.spherical_angles

        self.beamformer.operator = None
        with self.assertRaises(RuntimeError):
            _ = self.focus.spherical_angles

        self.focus.beamformer = None
        with self.assertRaises(RuntimeError):
            _ = self.focus.spherical_angles

    def test_spherical_angles(self) -> None:
        """Spherical angles property getter should return the correct angles"""

        expected_angles = Direction.From_Cartesian(-np.ones(3), normalize=True).to_spherical()
        assert_array_equal(expected_angles, self.focus.spherical_angles)


class TestCoordinateFocus(_TestBeamFocus):
    """Test coordinate focusing beam focus class."""

    def setUp(self) -> None:
        self.focus = CoordinateFocus(np.arange(3), "global")

        self.beamformer = Mock(spec=BeamformerBase)
        self.beamformer.operator = Mock()
        self.beamformer.operator.device = SimulatedDevice()
        self.focus.beamformer = self.beamformer

    def test_spherical_angles_validation(self) -> None:
        """Spherical angles property getter should raise RuntimeErrors on invalid configurations"""

        self.beamformer.operator.device = None
        with self.assertRaises(RuntimeError):
            _ = self.focus.spherical_angles

        self.beamformer.operator = None
        with self.assertRaises(RuntimeError):
            _ = self.focus.spherical_angles

        self.focus.beamformer = None
        with self.assertRaises(RuntimeError):
            _ = self.focus.spherical_angles

    def test_spherical_angles_local(self) -> None:
        """Spherical angles property getter should return the correct angles"""

        expected_direction = Direction.From_Cartesian(np.arange(3), normalize=True)
        self.focus = CoordinateFocus(expected_direction, "local")

        assert_array_equal(expected_direction.to_spherical(), self.focus.spherical_angles)

    def test_spherical_angles_global(self) -> None:
        expected_angles = Direction.From_Cartesian(np.arange(3), normalize=True).to_spherical()
        assert_array_equal(expected_angles, self.focus.spherical_angles)


class TestSphericalFocus(_TestBeamFocus):
    """Test spherical focusing beam focus class."""

    def setUp(self) -> None:
        self.focus = SphericalFocus(np.arange(2))

    def test_array_init(self) -> None:
        """Test initialization with a numpy array"""

        expected_angles = np.arange(2)
        focus = SphericalFocus(expected_angles)

        assert_array_equal(expected_angles, focus.spherical_angles)

    def test_scalar_init(self) -> None:
        """Test initialization with scalar elevation and azimuth"""

        excepted_angles = np.array([1, 2])
        focus = SphericalFocus(*excepted_angles)

        assert_array_equal(excepted_angles, focus.spherical_angles)

    def test_init_validation(self) -> None:
        """Initialization should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            SphericalFocus("wrong", "type")


class TestBeamformerBase(TestCase):
    """Test the base for all beamformers"""

    def setUp(self) -> None:
        self.operator = Mock()
        self.base = BeamformerBase(operator=self.operator)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.operator, self.base.operator)

    def test_operator_setget(self) -> None:
        """Operator property getter should return setter argument"""

        operator = Mock()
        self.base.operator = operator

        self.assertIs(operator, self.base.operator)


class TransmitBeamformerMock(TransmitBeamformer):
    """Mock class to test transmitting beamformers"""

    @property
    def num_transmit_input_streams(self) -> int:
        return 2

    @property
    def num_transmit_output_streams(self) -> int:
        return 2

    @property
    def num_transmit_focus_points(self) -> int:
        return 1

    def _encode(self, samples: np.ndarray, carrier_frequency: float, focus_angles: np.ndarray, array: AntennaArray) -> np.ndarray:
        return samples


class TestTransmitBeamformer(TestCase):
    def setUp(self) -> None:
        self.operator = Mock()
        self.operator.device = Mock()

        self.beamformer = TransmitBeamformerMock(operator=self.operator)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.operator, self.beamformer.operator)

    def test_encode_streams_validation(self) -> None:
        """Encode streams routine should raise exceptions on invalid arguments"""

        signal = Signal(np.zeros((3, 10), dtype=complex), 1.0)
        with self.assertRaises(ValueError):
            self.beamformer.encode_streams(signal)

    def test_encode_streams(self) -> None:
        """Stream encoding should properly encode the argument signal"""

        signal = Signal(np.ones((2, 10), dtype=complex), 1.0)
        encoded_signal = self.beamformer.encode_streams(signal)

        assert_array_equal(signal.samples, encoded_signal.samples)

    def test_precoding_setget(self) -> None:
        """Precoding property getter should return setter argument"""

        precoding = Mock()
        self.beamformer.precoding = precoding

        self.assertIs(precoding, self.beamformer.precoding)
        self.assertIs(precoding.modem, self.beamformer.operator)

    def test_transmit_focus_validation(self) -> None:
        """Transmit focus property setter should raise ValueError on invalid arguments"""

        focus_points = [Mock() for _ in range(1 + self.beamformer.num_transmit_focus_points)]
        with self.assertRaises(ValueError):
            self.beamformer.transmit_focus = focus_points

    def test_transmit_focus_setget(self) -> None:
        """Transmit focus property getters should return focus points property setter arguments"""

        expected_focus = [Mock(spec=BeamFocus) for _ in range(self.beamformer.num_transmit_focus_points)]

        self.beamformer.transmit_focus = expected_focus
        focus = self.beamformer.transmit_focus

        if not isinstance(focus, list):
            focus = [focus]

        self.assertEqual(len(expected_focus), len(focus))

        # Assert that the beamformer property has been set
        for f, fe in zip(focus, expected_focus):
            self.assertIs(self.beamformer, f.beamformer)
            self.assertEqual(f.spherical_angles, fe.spherical_angles)

    def test_transmit_validation(self) -> None:
        """Transmit routine should raise exceptions on invalid configurations"""

        with self.assertRaises(ValueError):
            self.beamformer.transmit(Signal(np.zeros((2, 10), dtype=complex), 1.0), [SphericalFocus(0, 0), SphericalFocus(1, 2)])

        with self.assertRaises(RuntimeError):
            self.beamformer.transmit(Signal(np.zeros((1, 10), dtype=complex), 1.0))

        self.operator.device = None

        with self.assertRaises(FloatingError):
            self.beamformer.transmit(Signal(np.zeros((2, 10), dtype=complex), 1.0))

        self.beamformer.operator = None

        with self.assertRaises(FloatingError):
            self.beamformer.transmit(Signal(np.zeros((2, 10), dtype=complex), 1.0))

    def test_transmit_focus_argument(self) -> None:
        """Transmit routine should correctly envoke the envode subroutine for scalar focus arguments"""

        expected_signal = Signal(np.ones((2, 10), dtype=complex), 1.0)
        focus = SphericalFocus(0, 0)

        steered_signal = self.beamformer.transmit(expected_signal, focus)
        assert_array_equal(expected_signal.samples, steered_signal.samples)

    def test_transmit_sequence_argument(self) -> None:
        """Transmit routine should correctly envoke the encode subroutine"""

        expected_signal = Signal(np.ones((2, 10), dtype=complex), 1.0)
        focus = [SphericalFocus(0, f) for f in range(self.beamformer.num_transmit_focus_points)]

        steered_signal = self.beamformer.transmit(expected_signal, focus)
        assert_array_equal(expected_signal.samples, steered_signal.samples)


class ReceiveBeamformerMock(ReceiveBeamformer):
    """Mock class to test receiving beamformers"""

    @property
    def num_receive_input_streams(self) -> int:
        return 2

    @property
    def num_receive_output_streams(self) -> int:
        return 2

    @property
    def num_receive_focus_points(self) -> int:
        return 1

    def _decode(self, samples: np.ndarray, carrier_frequency: float, angles: np.ndarray, array: AntennaArray) -> np.ndarray:
        return np.repeat(samples[np.newaxis, ::], angles.shape[0], axis=0)


class TestReceiveBeamformer(TestCase):
    def setUp(self) -> None:
        self.operator = Mock()
        self.operator.device = Mock()
        self.operator.device.carrier_frequency = 10e9
        self.operator.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (4, 4))

        self.beamformer = ReceiveBeamformerMock(operator=self.operator)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.operator, self.beamformer.operator)

    def test_decode_streams_validation(self) -> None:
        """Decode streams routine should raise exceptions on invalid arguments"""

        signal = Signal(np.zeros((3, 10), dtype=complex), 1.0)
        with self.assertRaises(ValueError):
            self.beamformer.decode_streams(signal)

    def test_decode_streams(self) -> None:
        """Stream decoding should properly encode the argument signal"""

        signal = Signal(np.ones((2, 10), dtype=complex), 1.0)
        decoded_signal = self.beamformer.decode_streams(signal)

        assert_array_equal(signal.samples, decoded_signal.samples)

    def test_precoding_setget(self) -> None:
        """Precoding property getter should return setter argument"""

        precoding = Mock()
        self.beamformer.precoding = precoding

        self.assertIs(precoding, self.beamformer.precoding)
        self.assertIs(precoding.modem, self.beamformer.operator)

    def test_receive_focus_validation(self) -> None:
        """Receive focus property setter should raise ValueError on invalid arguments"""

        focus_points = [Mock() for _ in range(1 + self.beamformer.num_receive_focus_points)]
        with self.assertRaises(ValueError):
            self.beamformer.receive_focus = focus_points

    def test_receive_focus_setget(self) -> None:
        """Receive focus property getters should return focus points property setter arguments"""

        expected_focus = [Mock(spec=BeamFocus) for _ in range(self.beamformer.num_receive_focus_points)]

        self.beamformer.receive_focus = expected_focus
        focus = self.beamformer.receive_focus

        if not isinstance(focus, list):
            focus = [focus]

        self.assertEqual(len(expected_focus), len(focus))

        # Assert that the beamformer property has been set
        for f, fe in zip(focus, expected_focus):
            self.assertIs(self.beamformer, f.beamformer)
            self.assertEqual(f.spherical_angles, fe.spherical_angles)

    def test_probe_focus_point_validation(self) -> None:
        """Focus point property setter should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 3))

        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 3, 4, 1))

        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 2))

    def test_probe_focus_setget(self) -> None:
        """Probe focus getter should return setter argument"""

        expected_points = np.array([[1, 2]], dtype=complex)
        self.beamformer.probe_focus_points = expected_points

        assert_array_equal(expected_points[np.newaxis, ::], self.beamformer.probe_focus_points)

    def test_receive_validation(self) -> None:
        """Receive routine should raise exceptions on invalid configurations"""

        with self.assertRaises(ValueError):
            self.beamformer.receive(Signal(np.zeros((2, 10), dtype=complex), 1.0), [SphericalFocus(0, 0), SphericalFocus(1, 2)])

        with self.assertRaises(RuntimeError):
            self.beamformer.receive(Signal(np.zeros((1, 10), dtype=complex), 1.0))

        self.operator.device = None

        with self.assertRaises(FloatingError):
            self.beamformer.receive(Signal(np.zeros((2, 10), dtype=complex), 1.0))

        self.beamformer.operator = None

        with self.assertRaises(FloatingError):
            self.beamformer.receive(Signal(np.zeros((2, 10), dtype=complex), 1.0))

    def test_receive_scalar_argument(self) -> None:
        """Receive routine should correctly envoke the envode subroutine for scalar focus arguments"""

        expected_signal = Signal(np.ones((2, 10), dtype=complex), 1.0)
        focus = SphericalFocus(0, 0)

        steered_signal = self.beamformer.receive(expected_signal, focus)
        assert_array_equal(expected_signal.samples, steered_signal.samples)

    def test_receive_sequence_argument(self) -> None:
        """Receive routine should correctly envoke the encode subroutine"""

        expected_signal = Signal(np.ones((2, 10), dtype=complex), 1.0)
        focus = [SphericalFocus(0, f) for f in range(self.beamformer.num_receive_focus_points)]

        steered_signal = self.beamformer.receive(expected_signal, focus)
        assert_array_equal(expected_signal.samples, steered_signal.samples)

    def test_probe_validation(self) -> None:
        """Probe routine should raise exceptions on invalid configurations"""

        with self.assertRaises(RuntimeError):
            self.beamformer.probe(Signal(np.zeros((1, 10), dtype=complex), 1.0))

        self.operator.device = None

        with self.assertRaises(FloatingError):
            self.beamformer.probe(Signal(np.zeros((2, 10), dtype=complex), 1.0))

        self.beamformer.operator = None

        with self.assertRaises(FloatingError):
            self.beamformer.probe(Signal(np.zeros((2, 10), dtype=complex), 1.0))

    def test_probe(self) -> None:
        """Probe routine should correctly envoke the encode subroutine"""

        expected_samples = np.ones((2, 10), dtype=complex)
        expected_signal = Signal(expected_samples, 1.0)
        focus = np.ones((1, 2, self.beamformer.num_receive_focus_points), dtype=float)

        steered_signal = self.beamformer.probe(expected_signal, focus)
        assert_array_equal(expected_samples[np.newaxis, ::], steered_signal)


del _TestBeamFocus
