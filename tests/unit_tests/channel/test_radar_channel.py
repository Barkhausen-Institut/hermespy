# -*- coding: utf-8 -*-
"""Test Radar Channel"""

from __future__ import annotations
from typing import Generic, TypeVar
import unittest
from unittest.mock import patch, Mock, PropertyMock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.constants import pi, speed_of_light

from hermespy.channel import SingleTargetRadarChannel, MultiTargetRadarChannel, VirtualRadarTarget, PhysicalRadarTarget, FixedCrossSection
from hermespy.channel.radar import VirtualRadarTarget, PhysicalRadarTarget
from hermespy.channel.radar.radar import RadarChannelBase, RadarPath, RadarTargetPath, RadarInterferencePath, RadarChannelRealization
from hermespy.core import ChannelStateInformation, ChannelStateFormat, Direction, Signal, Transformation
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.simulation.animation import Moveable, StaticTrajectory
from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import assert_signals_equal

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


RPT = TypeVar("RPT", bound=RadarPath)
RCRT = TypeVar("RCRT", bound=RadarChannelRealization)
RCT = TypeVar("RCT", bound=RadarChannelBase)


class TestFixedCrossSection(unittest.TestCase):
    """Test the fixed radar cross section model"""

    def setUp(self) -> None:
        self.cross_section = FixedCrossSection(1.23454)

    def test_init(self) -> None:
        """Class initialization parameters should be properly stored as attributes"""

        self.assertEqual(1.23454, self.cross_section.cross_section)

    def test_cross_section_setget(self) -> None:
        """Cross section property getter should return setter argument"""

        expected_cross_section = 2.34455
        self.cross_section.cross_section = expected_cross_section

        self.assertEqual(expected_cross_section, self.cross_section.cross_section)

    def test_cross_section_validation(self) -> None:
        """Cross section property setter should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.cross_section.cross_section = -1.0

    def test_get_cross_section(self) -> None:
        """Getting a cross section should return the fixed value"""

        impinging_direction = Direction.From_Spherical(0.0, 0.0)
        emerging_direction = Direction.From_Spherical(1.0, 1.0)

        cross_section = self.cross_section.get_cross_section(impinging_direction, emerging_direction)
        self.assertEqual(1.23454, cross_section)

    def test_serialization(self) -> None:
        """Test fixed cross section serialization"""

        test_roundtrip_serialization(self, self.cross_section)


class TestVirtualRadarTarget(unittest.TestCase):
    def setUp(self) -> None:
        self.cross_section = FixedCrossSection(1.234)
        self.velocity = np.array([1, 2, 3], dtype=float)
        self.pose = Transformation.From_Translation(np.array([2, 3, 4]))
        self.trajectory = StaticTrajectory(self.pose, self.velocity)

        self.target = VirtualRadarTarget(self.cross_section, self.trajectory)

    def test_init(self) -> None:
        """Initialization paramters should be properly stored as class attributes"""

        self.assertIs(self.cross_section, self.target.cross_section)
        self.assertFalse(self.target.static)

    def test_cross_section_setget(self) -> None:
        """Cross section property getter should return setter argument"""

        expected_cross_section = FixedCrossSection(2.345)
        self.target.cross_section = expected_cross_section

        self.assertIs(expected_cross_section, self.target.cross_section)

    def test_sample_cross_section(self) -> None:
        """Getting target parameters should return correct information"""

        self.assertEqual(1.234, self.target.sample_cross_section(Direction(np.array([1, 0, 0])), Direction(np.array([1, 0, 0]))))

    def test_sample_trajectory(self) -> None:
        """Getting target parameters should return correct information"""
        
        sample = self.target.sample_trajectory(0.0)
        self.assertEqual(0.0, sample.timestamp)
        

class TestPhysicalRadarTarget(unittest.TestCase):
    def setUp(self) -> None:
        self.cross_section = FixedCrossSection(1.234)
        self.velocity = np.array([1, 2, 3], dtype=float)
        self.pose = Transformation.From_Translation(np.array([2, 3, 4]))
        self.moveable = Moveable(StaticTrajectory(self.pose, self.velocity))

        self.target = PhysicalRadarTarget(self.cross_section, self.moveable)

    def test_init(self) -> None:
        """Initialization paramters should be properly stored as class attributes"""

        self.assertIs(self.cross_section, self.target.cross_section)
        self.assertIs(self.moveable, self.target.moveable)
        self.assertFalse(self.target.static)

    def test_cross_section_setget(self) -> None:
        """Cross section property getter should return setter argument"""

        expected_cross_section = FixedCrossSection(2.345)
        self.target.cross_section = expected_cross_section

        self.assertIs(expected_cross_section, self.target.cross_section)

    def test_sample_cross_section(self) -> None:
        """Sample target parameters should return correct information"""

        self.assertEqual(1.234, self.target.sample_cross_section(Direction(np.array([1, 0, 0])), Direction(np.array([1, 0, 0]))))

    def test_sample_trajectory(self) -> None:
        """Sample target parameters should return correct information"""

        sample = self.target.sample_trajectory(0.0)
        self.assertEqual(0.0, sample.timestamp)


class _TestRadarPathRealization(Generic[RPT], unittest.TestCase):
    """Test the radar path realization base class"""

    def _init_realization(self) -> RPT:
        ...

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.sampling_rate = 1e8
        self.carrier_frequency = 1e9

        self.attenuate = True
        self.static = False
        self.path_realization = self._init_realization()

        self.transmitter = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (2, 1, 1)), pose=Transformation.From_Translation(np.array([0.0, 0.0, 0.0], dtype=float)))
        self.receiver = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (2, 1, 1)), pose=Transformation.From_Translation(np.array([100.0, 0.0, 0.0], dtype=float)))

    def _test_propagate_state(self) -> None:
        test_signal = Signal.Create(self.rng.standard_normal((self.transmitter.antennas.num_transmit_antennas, 10)) + 1j * self.rng.standard_normal((self.transmitter.antennas.num_transmit_antennas, 10)), self.sampling_rate, self.carrier_frequency)

        expected_sample_offset = int(self.path_realization.propagation_delay(self.transmitter, self.receiver) * self.sampling_rate)
        propagated_samples = np.zeros((self.receiver.antennas.num_receive_antennas, test_signal.num_samples + expected_sample_offset), dtype=np.complex128)
        self.path_realization.add_propagation(self.transmitter.state(0), self.receiver.state(0), test_signal.getitem(), test_signal.sampling_rate, test_signal.carrier_frequency, propagated_samples)

        raw_state = np.zeros((self.receiver.antennas.num_receive_antennas, self.transmitter.antennas.num_transmit_antennas, test_signal.num_samples, 1 + expected_sample_offset), dtype=np.complex128)
        self.path_realization.add_state(self.transmitter.state(0), self.receiver.state(0), self.sampling_rate, self.carrier_frequency, 0.0, raw_state)
        channel_state = ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, raw_state)
        state_propagated_samples = channel_state.propagate(test_signal).getitem()

        assert_array_almost_equal(propagated_samples, state_propagated_samples[:, :propagated_samples.size])

    def test_attenuate_setget(self) -> None:
        """Attenuate property getter should return setter argument"""

        self.path_realization.attenuate = False
        self.assertFalse(self.path_realization.attenuate)

    def test_static_setget(self) -> None:
        """Static property getter should return setter argument"""

        self.path_realization.static = False
        self.assertFalse(self.path_realization.static)

    def test_propagtate_state(self) -> None:
        """Propagation and state should be equivalent"""

        self.path_realization.attenuate = False
        self._test_propagate_state()

    def test_propagate_state_attenuate(self) -> None:
        """Propagation with attenuation and attenuated state should be equivalent"""

        self.path_realization.attenuate = True
        self._test_propagate_state()

    def test_add_state_out_of_range_delay(self) -> None:
        """Adding a delayed state with a too high delay should do nothing"""

        state = np.zeros((1, 1, 5, 10), dtype=np.complex128)
        self.path_realization.add_state(self.transmitter.state(0), self.receiver.state(0), self.sampling_rate, self.carrier_frequency, -1e10, state)
        assert_array_equal(np.zeros_like(state), state)

        state = np.zeros((1, 1, 5, 1), dtype=np.complex128)
        self.path_realization.add_state(self.transmitter.state(0), self.receiver.state(0), self.sampling_rate, self.carrier_frequency, 0.0, state)
        assert_array_equal(np.zeros_like(state), state)

    def test_serialization(self) -> None:
        """Test radar path serialization"""
        
        test_roundtrip_serialization(self, self.path_realization)


class TestRadarTargetRealization(_TestRadarPathRealization[RadarTargetPath]):
    """Test the radar target realization class"""

    def _init_realization(self) -> RadarTargetPath:
        self.target_position = np.array([1.0, 2.0, 3.0], dtype=float)
        self.target_velocity = np.array([4.0, 5.0, 6.0], dtype=float)
        self.cross_section = 1.234
        self.reflection_phase = -1

        return RadarTargetPath(self.target_position, self.target_velocity, self.cross_section, self.reflection_phase, self.attenuate, self.static)

    def test_properties(self) -> None:
        """Class properties should return initialization arguments"""

        assert_array_equal(self.target_position, self.path_realization.position)
        assert_array_equal(self.target_velocity, self.path_realization.velocity)
        self.assertEqual(self.cross_section, self.path_realization.cross_section)
        self.assertEqual(self.reflection_phase, self.path_realization.reflection_phase)


class TestInterferenceRealization(_TestRadarPathRealization[RadarInterferencePath]):
    """Test the radar interference realization class"""

    def _init_realization(self) -> RadarInterferencePath:
        return RadarInterferencePath(self.attenuate, self.static)


class _TestRadarChannelBase(Generic[RCT], unittest.TestCase):
    def _init_channel(self) -> RCT:
        ...  # pragma: no cover

    def setUp(self) -> None:
        self.random_generator = default_rng(42)
        self.random_root = Mock()
        self.random_root._rng = self.random_generator

        self.sampling_rate = 1e6
        self.carrier_frequency = 1e9

        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (1, 1, 1)))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.01, (1, 1, 1)))

        self.channel = self._init_channel()
        self.channel.random_mother = self.random_root

    def test_attenuate_setget(self) -> None:
        """Attenuate property getter should return setter argument"""

        self.channel.attenuate = False
        self.assertFalse(self.channel.attenuate)

    def test_model_serialization(self) -> None:
        """Test radar channel model serialization"""

        test_roundtrip_serialization(self, self.channel, {'random_mother'})

    def test_realization_serialization(self) -> None:
        """Test radar channel realization serialization"""

        test_roundtrip_serialization(self, self.channel.realize())

    def test_propagate_state(self) -> None:
        """Test if the state propagation is correct"""
        
        test_signal = Signal.Create(self.random_generator.standard_normal((self.alpha_device.antennas.num_transmit_antennas, 10)) + 1j * self.random_generator.standard_normal((self.alpha_device.antennas.num_transmit_antennas, 10)), self.sampling_rate, self.carrier_frequency)

        realization = self.channel.realize()
        sample = realization.sample(self.alpha_device, self.beta_device)
        
        sample_propagation = sample.propagate(test_signal)
        state_propagation = sample.state(10, 2).propagate(test_signal)

        assert_signals_equal(self, sample_propagation, state_propagation)
        
    
class TestSingleTargetRadarChannel(_TestRadarChannelBase[SingleTargetRadarChannel]):
    def _init_channel(self) -> SingleTargetRadarChannel:
        return SingleTargetRadarChannel(self.range, self.radar_cross_section)

    def setUp(self) -> None:
        self.range = 100.0
        self.radar_cross_section = 1.0
        self.expected_delay = 2 * self.range / speed_of_light

        super().setUp()

    def test_target_range_setget(self) -> None:
        """Target range property getter should return setter argument"""

        new_range = 500
        self.channel.target_range = new_range

        self.assertEqual(new_range, self.channel.target_range)

    def test_target_range_validation(self) -> None:
        """Target range property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.target_range = -1.0

        with self.assertRaises(ValueError):
            self.channel.target_range = (1, 2, 3)

        with self.assertRaises(ValueError):
            self.channel.target_range = (3, 2)

        with self.assertRaises(ValueError):
            self.channel.target_range = (-1, 0)

        with self.assertRaises(ValueError):
            self.channel.target_range = "wrong argument type"
            
    def test_target_velocity_setget(self) -> None:
        """Target velocity property getter should return setter argument"""

        new_velocity = 10
        self.channel.target_velocity = new_velocity
        self.assertEqual(new_velocity, self.channel.target_velocity)
        
        new_velocity = (0, 10)
        self.channel.target_velocity = new_velocity
        self.assertEqual(new_velocity, self.channel.target_velocity)
        
        new_velocity = np.array([1, 2, 3])
        self.channel.target_velocity = new_velocity
        assert_array_equal(new_velocity, self.channel.target_velocity)
        
    def test_target_velocity_validation(self) -> None:
        """Target velocity property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.target_velocity = (1, 2, 3, 4)

        with self.assertRaises(ValueError):
            self.channel.target_velocity = (3, 2)

    def test_radar_cross_section_get(self) -> None:
        """Radar cross section getter should return init param"""

        self.assertEqual(self.radar_cross_section, self.channel.radar_cross_section)

    def test_cross_section_validation(self) -> None:
        """Radar cross section property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.channel.radar_cross_section = -1.12345

        try:
            self.channel.radar_cross_section = 0.0

        except ValueError:
            self.fail()
            
    def test_target_azimuth_setget(self) -> None:
        """Target azimuth property getter should return setter argument"""

        new_azimuth = 10
        self.channel.target_azimuth = new_azimuth
        self.assertEqual(new_azimuth, self.channel.target_azimuth)
        
        new_azimuth = (0, 10)
        self.channel.target_azimuth = new_azimuth
        self.assertEqual(new_azimuth, self.channel.target_azimuth)
        
    def test_target_azimuth_validation(self) -> None:
        """Target azimuth property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.target_azimuth = (1, 2, 3, 4)

        with self.assertRaises(ValueError):
            self.channel.target_azimuth = (3, 2)
            
    def test_target_zenith_setget(self) -> None:
        """Target zenith property getter should return setter argument"""

        new_zenith = 10
        self.channel.target_zenith = new_zenith
        self.assertEqual(new_zenith, self.channel.target_zenith)
        
        new_zenith = (0, 10)
        self.channel.target_zenith = new_zenith
        self.assertEqual(new_zenith, self.channel.target_zenith)
        
    def test_target_zenith_validation(self) -> None:
        """Target zenith property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.target_zenith = (1, 2, 3, 4)

        with self.assertRaises(ValueError):
            self.channel.target_zenith = (3, 2)

    def test_target_exists_setget(self) -> None:
        """Target exists flag getter should return setter argument"""

        new_target_exists = False
        self.channel.target_exists = new_target_exists
        self.assertEqual(new_target_exists, self.channel.target_exists)

    def _create_impulse_train(self, interval_in_samples: int, number_of_pulses: int):
        interval = interval_in_samples / self.sampling_rate

        number_of_samples = int(np.ceil(interval * self.sampling_rate * number_of_pulses))
        output_signal = np.zeros((1, number_of_samples), dtype=complex)

        interval_in_samples = int(np.around(interval * self.sampling_rate))

        output_signal[:, :number_of_samples:interval_in_samples] = 1.0

        return output_signal
    
    def test_decorrelation_distance_setget(self) -> None:
        """Decorrelation distance property getter should return setter argument"""

        new_decorrelation_distance = 10
        self.channel.decorrelation_distance = new_decorrelation_distance
        self.assertEqual(new_decorrelation_distance, self.channel.decorrelation_distance)
        
    def test_decorrelation_distance_validation(self) -> None:
        """Decorrelation distance property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.decorrelation_distance = -1.12345

    def test_propagation_delay_integer_num_samples(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay is a multiple
        of the sampling interval
        """

        samples_per_symbol = 1000
        num_pulses = 10
        delay_in_samples = 507

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = speed_of_light * delay_in_samples / self.sampling_rate / 2
        expected_amplitude = (speed_of_light / self.carrier_frequency) ** 2 * self.radar_cross_section / (4 * pi) ** 3 / expected_range**4

        self.channel.target_range = expected_range

        propagation = self.channel.propagate(Signal.Create(input_signal, self.sampling_rate, self.carrier_frequency), self.alpha_device, self.beta_device)

        expected_output = np.hstack((np.zeros((1, delay_in_samples)), input_signal)) * expected_amplitude
        assert_array_almost_equal(abs(expected_output), np.abs(propagation.getitem((slice(None, None), slice(None, expected_output.size)))))

    def test_propagation_delay_noninteger_num_samples(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay falls in the
        middle of two sampling instants.
        """
        samples_per_symbol = 800
        num_pulses = 20
        delay_in_samples = 312

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = speed_of_light * (delay_in_samples + 0.5) / self.sampling_rate / 2
        expected_amplitude = (speed_of_light / self.carrier_frequency) ** 2 * self.radar_cross_section / (4 * pi) ** 3 / expected_range**4

        self.channel.target_range = expected_range

        propagation = self.channel.propagate(Signal.Create(input_signal, self.sampling_rate, self.carrier_frequency), self.alpha_device, self.beta_device)

        straddle_loss = np.sinc(0.5)
        peaks = np.abs(propagation.getitem((slice(None, None), slice(delay_in_samples, input_signal.size, samples_per_symbol))))

        assert_array_almost_equal(peaks, expected_amplitude * straddle_loss * np.ones(peaks.shape))

    def test_propagation_delay_doppler(self) -> None:
        """
        Test if the received signal corresponds to a frequency-shifted version of the transmitted signal with the
        expected Doppler shift
        """

        samples_per_symbol = 50
        num_pulses = 100
        initial_delay_in_samples = 1000
        expected_range = speed_of_light * initial_delay_in_samples / self.sampling_rate / 2
        velocity = 10
        expected_amplitude = (speed_of_light / self.carrier_frequency) ** 2 * self.radar_cross_section / (4 * pi) ** 3 / expected_range**4

        initial_delay = initial_delay_in_samples / self.sampling_rate

        timestamps_impulses = np.arange(num_pulses) * samples_per_symbol / self.sampling_rate
        traveled_distances = velocity * timestamps_impulses
        delays = initial_delay + 2 * traveled_distances / speed_of_light
        expected_peaks = timestamps_impulses + delays
        peaks_in_samples = np.around(expected_peaks * self.sampling_rate).astype(int)
        straddle_delay = expected_peaks - peaks_in_samples / self.sampling_rate
        relative_straddle_delay = straddle_delay * self.sampling_rate
        expected_straddle_amplitude = np.sinc(relative_straddle_delay) * expected_amplitude

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.target_range = expected_range
        self.channel.target_velocity = velocity

        propagation = self.channel.propagate(Signal.Create(input_signal, self.sampling_rate, self.carrier_frequency), self.alpha_device, self.beta_device)

        assert_array_almost_equal(np.abs(propagation.getitem()[0, peaks_in_samples].flatten()), expected_straddle_amplitude)

    def test_propagation_random_paramters(self) -> None:
        """Test target parameter sampling from intervals"""
        
        self.channel.target_range = (10, 20)
        self.channel.target_azimuth = (-pi, pi)
        self.channel.target_zenith = (0, pi)
        self.channel.target_velocity = (0, 10)
        
        sample = self.channel.realize().sample(self.alpha_device, self.beta_device)
        target_path = sample.paths[0]
        
        self.assertTrue(10 <= .5 * target_path.propagation_delay(self.alpha_device.state(0), self.beta_device.state(0)) * speed_of_light <= 20)
        
    def test_doppler_shift(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given time variant delays on account of
        movement
        """

        velocity = 100
        self.channel.target_velocity = velocity

        num_samples = 100000
        sinewave_frequency = 0.25 * self.sampling_rate
        doppler_shift = 2 * velocity / speed_of_light * self.carrier_frequency

        time = np.arange(num_samples) / self.sampling_rate

        input_signal = np.sin(2 * np.pi * sinewave_frequency * time)
        propagation = self.channel.propagate(Signal.Create(input_signal[np.newaxis, :], self.sampling_rate, self.carrier_frequency), self.alpha_device, self.beta_device)

        input_freq = np.fft.fft(input_signal)
        output_freq = np.fft.fft(propagation.getitem((0, slice(-num_samples, None))).flatten())

        freq_resolution = self.sampling_rate / num_samples

        freq_in = np.argmax(np.abs(input_freq[: int(num_samples / 2)])) * freq_resolution
        freq_out = np.argmax(np.abs(output_freq[: int(num_samples / 2)])) * freq_resolution

        self.assertAlmostEqual(freq_out - freq_in, doppler_shift, delta=np.abs(doppler_shift) * 0.01)

    def test_no_echo(self) -> None:
        """Test if no echos are observed if target_exists flag is disabled"""

        samples_per_symbol = 500
        num_pulses = 15

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.target_exists = False
        propagation = self.channel.propagate(Signal.Create(input_signal, self.sampling_rate), self.alpha_device, self.beta_device)

        assert_array_almost_equal(propagation.getitem(), np.zeros_like(input_signal))

    def test_no_attenuation(self) -> None:
        """Make sure the signal energy is preserved when the attenuate flag is disabled"""

        self.channel.attenuate = False
        self.channel.target_range = 10.0

        input_signal = Signal.Create(self._create_impulse_train(500, 15), self.sampling_rate)
        propagation = self.channel.propagate(input_signal, self.alpha_device, self.beta_device)

        assert_array_almost_equal(input_signal.energy, propagation.energy, 1)


class TestMultiTargetRadarChannel(_TestRadarChannelBase[MultiTargetRadarChannel]):
    """Test the multi target radar channel class"""

    def _init_channel(self) -> MultiTargetRadarChannel:
        return MultiTargetRadarChannel()

    def setUp(self) -> None:
        super().setUp()

        self.first_target = VirtualRadarTarget(FixedCrossSection(1.0), trajectory=StaticTrajectory(Transformation.From_Translation(np.array([-10, 0, 0], dtype=float)), velocity=np.array([10, 0, 0])))
        self.second_target = VirtualRadarTarget(FixedCrossSection(1.0), trajectory=StaticTrajectory(Transformation.From_Translation(np.array([10, 0, 0], dtype=float)), velocity=np.array([-10, 0, 0])))

        self.channel.add_target(self.first_target)
        self.channel.add_target(self.second_target)
        
    def test_decorrelation_distance_setget(self) -> None:
        """Decorrelation distance property getter should return setter argument"""

        new_decorrelation_distance = 10.0
        self.channel.decorrelation_distance = new_decorrelation_distance

        self.assertEqual(new_decorrelation_distance, self.channel.decorrelation_distance)
        
    def test_decorrelation_distance_validation(self) -> None:
        """Decorrelation distance property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.decorrelation_distance = -1.0

    def test_add_virtual_target(self) -> None:
        """Test adding a new virtual radar target to the channel"""

        target = VirtualRadarTarget(FixedCrossSection(1.0))
        self.channel.add_target(target)

        self.assertTrue(target in self.channel.targets)

    def test_add_physical_target(self) -> None:
        """Test adding a new physical radar target to the channel"""

        target = PhysicalRadarTarget(FixedCrossSection(1.0), SimulatedDevice())
        self.channel.add_target(target)

        self.assertTrue(target in self.channel.targets)

    def test_make_target(self) -> None:
        """Test declaring a moveable as a radar target"""

        moveable = Moveable(StaticTrajectory(Transformation.From_Translation(np.array([0, 0, 0], dtype=float)), np.array([0, 0, 0])))
        crosse_section = FixedCrossSection(1.1234)
        new_target = self.channel.make_target(moveable, crosse_section)

        self.assertCountEqual([self.first_target, self.second_target, new_target], self.channel.targets)

    def test_target_device_collision(self) -> None:
        """Sampling should fail if a device collides with a target"""
        
        self.alpha_device.trajectory = self.first_target.trajectory
        
        with self.assertRaises(RuntimeError):
            self.channel.realize().sample(self.alpha_device, self.beta_device)

        with self.assertRaises(RuntimeError):
            self.channel.realize().sample(self.beta_device, self.alpha_device)

    def test_interference(self) -> None:
        """Interference path should be properly added during sampling"""
        
        self.alpha_device.trajectory = StaticTrajectory(Transformation.From_Translation(np.array([0, 0, 0], dtype=float)), np.array([0, 0, 0]))
        self.beta_device.trajectory = StaticTrajectory(Transformation.From_Translation(np.array([123, 0, 0], dtype=float)), np.array([0, 0, 0]))
        
        self.channel.interference = False
        non_interference_sample = self.channel.realize().sample(self.alpha_device, self.beta_device)
        
        self.channel.interference = True
        interference_sample = self.channel.realize().sample(self.alpha_device, self.beta_device)
        
        self.assertGreater(len(interference_sample.paths), len(non_interference_sample.paths))


del _TestRadarPathRealization
del _TestRadarChannelBase
