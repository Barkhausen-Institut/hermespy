# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.beamforming import BeamFocus, CoordinateFocus, ReceiveBeamformer, SphericalFocus, TransmitBeamformer
from hermespy.core import AntennaArray, DenseSignal, DeviceState, Direction, Device, FloatingError, DenseSignal, Transformation
from hermespy.simulation import DeviceFocus, SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import assert_signals_equal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockBeamFocus(BeamFocus):
    """Mock class to test the beam focus base class"""

    def spherical_angles(self, device: DeviceState) -> np.ndarray:
        return np.arange(2)

    def copy(self) -> BeamFocus:
        return MockBeamFocus()


class TestBeamFocus(TestCase):
    """Test the beam focus base class"""

    def setUp(self) -> None:

        self.focus = MockBeamFocus()


class _TestBeamFocus(TestCase):
    """Test classes inheriting from the beam focus base class"""

    device_focus: BeamFocus

    def test_copy(self) -> None:
        """Copy routine should return a copy of the beam focus"""

        copy = self.device_focus.copy()

        self.assertIsNot(copy, self.device_focus)

    def test_serialization(self) -> None:
        """Test beam focus serialization"""

        test_roundtrip_serialization(self, self.device_focus)


class TestDeviceFocus(_TestBeamFocus):
    """Test device focusing beam focus class."""

    def setUp(self) -> None:
        self.focusing_device = SimulatedDevice()
        self.focused_device = SimulatedDevice(
            pose=Transformation.From_Translation(np.ones(3))
        )

        self.device_focus = DeviceFocus(self.focused_device)

    def test_spherical_angles(self) -> None:
        """Spherical angles property getter should return the correct angles"""

        expected_angles = Direction.From_Cartesian(np.ones(3), normalize=True).to_spherical()
        assert_array_equal(expected_angles, self.device_focus.spherical_angles(self.focusing_device.state()))


class TestCoordinateFocus(_TestBeamFocus):
    """Test coordinate focusing beam focus class."""

    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.device_focus = CoordinateFocus(np.arange(3), "global")

    def test_spherical_angles_local(self) -> None:
        """Spherical angles property getter should return the correct angles"""

        expected_direction = Direction.From_Cartesian(np.arange(3), normalize=True)
        self.device_focus = CoordinateFocus(expected_direction, "local")

        assert_array_equal(expected_direction.to_spherical(), self.device_focus.spherical_angles(self.device.state()))

    def test_spherical_angles_global(self) -> None:
        expected_angles = Direction.From_Cartesian(np.arange(3), normalize=True).to_spherical()
        assert_array_equal(expected_angles, self.device_focus.spherical_angles(self.device.state()))


class TestSphericalFocus(_TestBeamFocus):
    """Test spherical focusing beam focus class."""

    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.device_focus = SphericalFocus(np.arange(2))

    def test_array_init(self) -> None:
        """Test initialization with a numpy array"""

        expected_angles = np.arange(2)
        focus = SphericalFocus(expected_angles)

        assert_array_equal(expected_angles, focus.spherical_angles(self.device.state()))

    def test_scalar_init(self) -> None:
        """Test initialization with scalar elevation and azimuth"""

        excepted_angles = np.array([1, 2])
        focus = SphericalFocus(*excepted_angles)

        assert_array_equal(excepted_angles, focus.spherical_angles(self.device.state()))

    def test_init_validation(self) -> None:
        """Initialization should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            SphericalFocus("wrong", "type")


class TransmitBeamformerMock(TransmitBeamformer):
    """Mock class to test transmitting beamformers"""

    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        return num_output_streams

    @property
    def num_transmit_focus_points(self) -> int:
        return 1

    def _encode(
        self,
        samples: np.ndarray,
        carrier_frequency: float,
        focus_angles: np.ndarray,
        array: AntennaArray,
    ) -> np.ndarray:
        return samples


class TestTransmitBeamformer(TestCase):
    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.beamformer = TransmitBeamformerMock()

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

    def test_encode_streams_validation(self) -> None:
        """Encode streams routine should raise exceptions on invalid arguments"""

        # Correct number of input signal streams
        signal = DenseSignal.FromNDArray(np.zeros((3, 10), dtype=complex), 1.0)
        with self.assertRaises(ValueError):
            self.beamformer.encode_streams(signal, 2, self.device.state().transmit_state())

        # Incorrect number of focus points
        signal = DenseSignal.FromNDArray(np.zeros((2, 10), dtype=complex), 1.0)
        with self.assertRaises(ValueError):
            self.beamformer.encode_streams(signal, 2, self.device.state().transmit_state(), [SphericalFocus(0, 0), SphericalFocus(1, 2)])

    def test_encode_streams_focus(self) -> None:
        """The focus should be correctly processed in the encode subroutine"""

        signal = DenseSignal.FromNDArray(np.ones((2, 10), dtype=complex), 1.0)
        self.beamformer.transmit_focus = SphericalFocus(1, 2)

        encoded_signal_no_focus = self.beamformer.encode_streams(signal, 2, self.device.state().transmit_state(), None)
        encoded_signal_simple_focus = self.beamformer.encode_streams(signal, 2, self.device.state().transmit_state(), SphericalFocus(1, 2))
        encode_signal_list_focus = self.beamformer.encode_streams(signal, 2, self.device.state().transmit_state(), [SphericalFocus(1, 2)])

        assert_signals_equal(self, signal, encoded_signal_no_focus)
        assert_signals_equal(self, signal, encoded_signal_simple_focus)
        assert_signals_equal(self, signal, encode_signal_list_focus)

    def test_encode_streams(self) -> None:
        """Stream encoding should properly encode the argument signal"""

        signal = DenseSignal.FromNDArray(np.ones((2, 10), dtype=complex), 1.0)
        encoded_signal = self.beamformer.encode_streams(signal, 2, self.device.state().transmit_state())

        assert_signals_equal(self, signal, encoded_signal)


class ReceiveBeamformerMock(ReceiveBeamformer):
    """Mock class to test receiving beamformers"""

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        return 1

    @property
    def num_receive_focus_points(self) -> int:
        return 1

    def _decode(
        self, samples: np.ndarray, carrier_frequency: float, angles: np.ndarray, array: AntennaArray
    ) -> np.ndarray:
        return np.repeat(samples[np.newaxis, ::], angles.shape[0], axis=0)


class TestReceiveBeamformer(TestCase):
    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.beamformer = ReceiveBeamformerMock()

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

    def test_probe_focus_points_validation(self) -> None:
        """Focus point property setter should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 3))

        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 3, 4, 1))

        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 2))

    def test_probe_focus_points_setget(self) -> None:
        """Probe focus getter should return setter argument"""

        expected_points = np.array([[1, 2]], dtype=complex)
        self.beamformer.probe_focus_points = expected_points

        assert_array_equal(expected_points[np.newaxis, ::], self.beamformer.probe_focus_points)

    def test_decode_streams_validation(self) -> None:
        """Decode streams routine should raise exceptions on invalid arguments"""

        # Invalid number of focus points
        signal = DenseSignal.FromNDArray(np.zeros((2, 10), dtype=complex), 1.0)
        with self.assertRaises(ValueError):
            self.beamformer.decode_streams(signal, 2, self.device.state().receive_state(), [SphericalFocus(0, 0), SphericalFocus(1, 2)])

    def test_decode_streams(self) -> None:
        """Stream decoding should properly encode the argument signal"""

        signal = DenseSignal.FromNDArray(np.ones((2, 10), dtype=complex), 1.0)
        decoded_signal = self.beamformer.decode_streams(signal, 2, self.device.state().receive_state())

        assert_signals_equal(self, signal, decoded_signal)

    def test_decode_streams_focus(self) -> None:
        """The focus should be correctly processed in the decode subroutine"""

        signal = DenseSignal.FromNDArray(np.ones((2, 10), dtype=complex), 1.0)
        self.beamformer.receive_focus = SphericalFocus(1, 2)

        decoded_signal_no_focus = self.beamformer.decode_streams(signal, 2, self.device.state().receive_state(), None)
        decoded_signal_simple_focus = self.beamformer.decode_streams(signal, 2, self.device.state().receive_state(), SphericalFocus(1, 2))
        decoded_signal_list_focus = self.beamformer.decode_streams(signal, 2, self.device.state().receive_state(), [SphericalFocus(1, 2)])

        assert_signals_equal(self, signal, decoded_signal_no_focus)
        assert_signals_equal(self, signal, decoded_signal_simple_focus)
        assert_signals_equal(self, signal, decoded_signal_list_focus)

    def test_probe(self) -> None:
        """Probe routine should correctly envoke the encode subroutine"""

        expected_samples = np.ones((2, 10), dtype=complex)
        focus = np.ones((1, 2, self.beamformer.num_receive_focus_points), dtype=float)

        steered_samples = self.beamformer.probe(expected_samples, self.device.state().receive_state(), focus)
        assert_array_equal(expected_samples[np.newaxis, ::], steered_samples)

del _TestBeamFocus
