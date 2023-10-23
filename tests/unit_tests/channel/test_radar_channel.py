# -*- coding: utf-8 -*-
"""Test Radar Channel"""

from __future__ import annotations
from typing import Generic, TypeVar
import unittest
from unittest.mock import patch, Mock, PropertyMock
from h5py import File

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.constants import pi, speed_of_light

from hermespy.channel import SingleTargetRadarChannel, MultiTargetRadarChannel, VirtualRadarTarget, PhysicalRadarTarget, FixedCrossSection
from hermespy.channel.radar_channel import MultiTargetRadarChannelRealization, SingleTargetRadarChannelRealization, RadarChannelBase, RadarChannelRealization, RadarPathRealization, RadarInterferenceRealization, RadarTargetRealization
from hermespy.core import ChannelStateInformation, ChannelStateFormat, Direction, FloatingError, IdealAntenna, Moveable, Signal, Transformation, UniformArray
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


RPRT = TypeVar('RPRT', bound=RadarPathRealization)
RCRT = TypeVar('RCRT', bound=RadarChannelRealization)
RCT = TypeVar('RCT', bound=RadarChannelBase)


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
            self.cross_section.cross_section = -1.
            
    def test_get_cross_section(self) -> None:
        """Getting a cross section should return the fixed value"""
        
        impinging_direction = Direction.From_Spherical(0., 0.)
        emerging_direction = Direction.From_Spherical(1., 1.)
        
        cross_section = self.cross_section.get_cross_section(impinging_direction, emerging_direction)
        self.assertEqual(1.23454, cross_section)

        
class TestVirtualRadarTarget(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.cross_section = FixedCrossSection(1.234)
        self.velocity = np.array([1, 2, 3], dtype=float)
        self.pose = Transformation.From_Translation(np.array([2, 3, 4]))
        
        self.target = VirtualRadarTarget(self.cross_section, self.velocity, self.pose)
        
    def test_init(self) -> None:
        """Initialization paramters should be properly stored as class attributes"""
        
        self.assertIs(self.cross_section, self.target.cross_section)
        assert_array_almost_equal(self.velocity, self.target.velocity)
        assert_array_almost_equal(self.pose, self.target.pose)
        
    def test_cross_section_setget(self) -> None:
        """Cross section property getter should return setter argument"""
        
        expected_cross_section = FixedCrossSection(2.345)
        self.target.cross_section = expected_cross_section
        
        self.assertIs(expected_cross_section, self.target.cross_section)
        
    def test_velocity_setget(self) -> None:
        """Velocity property getter should return setter argument"""
        
        expected_velocity = np.array([4, 5, 6])
        self.target.velocity = expected_velocity

        assert_array_almost_equal(expected_velocity, self.target.velocity)
        
    def test_get(self) -> None:
        """Getting target parameters should return correct information"""
        
        self.assertEqual(1.234, self.target.get_cross_section(Direction(np.array([1, 0, 0])), Direction(np.array([1, 0, 0]))))
        assert_array_almost_equal(self.velocity, self.target.get_velocity())
        assert_array_almost_equal(self.target.forwards_transformation, self.target.get_forwards_transformation())
        assert_array_almost_equal(self.target.backwards_transformation, self.target.get_backwards_transformation())


class TestPhysicalRadarTarget(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.cross_section = FixedCrossSection(1.234)
        self.velocity = np.array([1, 2, 3], dtype=float)
        self.pose = Transformation.From_Translation(np.array([2, 3, 4]))
        self.moveable = Moveable(self.pose, self.velocity)

        self.target = PhysicalRadarTarget(self.cross_section, self.moveable)

    def test_init(self) -> None:
        """Initialization paramters should be properly stored as class attributes"""

        self.assertIs(self.cross_section, self.target.cross_section)
        self.assertIs(self.moveable, self.target.moveable)

    def test_cross_section_setget(self) -> None:
        """Cross section property getter should return setter argument"""
        
        expected_cross_section = FixedCrossSection(2.345)
        self.target.cross_section = expected_cross_section
        
        self.assertIs(expected_cross_section, self.target.cross_section)
        
    def test_get(self) -> None:
        """Getting target parameters should return correct information"""
        
        self.assertEqual(1.234, self.target.get_cross_section(Direction(np.array([1, 0, 0])), Direction(np.array([1, 0, 0]))))
        assert_array_almost_equal(self.velocity, self.target.get_velocity())
        assert_array_almost_equal(self.moveable.forwards_transformation, self.target.get_forwards_transformation())
        assert_array_almost_equal(self.moveable.backwards_transformation, self.target.get_backwards_transformation())


class _TestRadarPathRealization(Generic[RPRT], unittest.TestCase):
    """Test the radar path realization base class"""

    def _init_realization(self) -> RPRT:
        ...
        
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        self.sampling_rate = 1e8
        self.carrier_frequency = 1e9
        
        self.attenuate = True
        self.static = False
        self.path_realization = self._init_realization()
        
        self.transmitter = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (2, 1, 1)), pose=Transformation.From_Translation(np.array([0., 0., 0.], dtype=float)))
        self.receiver = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (2, 1, 1)), pose=Transformation.From_Translation(np.array([100., 0., 0.], dtype=float)))
        
    def _test_propagate_state(self) -> None:
        
        test_signal = Signal(self.rng.standard_normal((self.transmitter.antennas.num_transmit_antennas, 10)) + 1j * self.rng.standard_normal((self.transmitter.antennas.num_transmit_antennas, 10)), self.sampling_rate, self.carrier_frequency)
        
        expected_sample_offset = int(self.path_realization.propagation_delay(self.transmitter, self.receiver) * self.sampling_rate)
        propagated_samples = np.zeros((self.receiver.antennas.num_receive_antennas, test_signal.num_samples + expected_sample_offset), dtype=np.complex_)
        self.path_realization.add_propagation(self.transmitter, self.receiver, test_signal, propagated_samples)
        
        raw_state = np.zeros((self.receiver.antennas.num_receive_antennas, self.transmitter.antennas.num_transmit_antennas, test_signal.num_samples, 1 + expected_sample_offset), dtype=np.complex_)
        self.path_realization.add_state(self.transmitter, self.receiver, 0, self.sampling_rate, raw_state)
        channel_state = ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, raw_state)
        state_propagated_samples = channel_state.propagate(test_signal).samples
        
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
        
        state = np.zeros((1, 1, 5, 10), dtype=np.complex_)
        
        self.path_realization.add_state(self.transmitter, self.receiver, -1e10, self.sampling_rate, state)
        
        assert_array_equal(np.zeros_like(state), state)


class TestRadarTargetRealization(_TestRadarPathRealization[RadarTargetRealization]):
    """Test the radar target realization class"""
    
    def _init_realization(self) -> RadarTargetRealization:
            
        self.target_position = np.array([1., 2., 3.], dtype=float)
        self.target_velocity = np.array([4., 5., 6.], dtype=float)
        self.cross_section = 1.234
        self.reflection_phase = -1

        return RadarTargetRealization(self.target_position, self.target_velocity, self.cross_section, self.reflection_phase, self.attenuate, self.static)

    def test_properties(self) -> None:
        """Class properties should return initialization arguments"""
        
        assert_array_equal(self.target_position, self.path_realization.position)
        assert_array_equal(self.target_velocity, self.path_realization.velocity)
        self.assertEqual(self.cross_section, self.path_realization.cross_section)
        self.assertEqual(self.reflection_phase, self.path_realization.reflection_phase)

    def test_hdf_serialization(self) -> None:
        """Test serialization to and from HDF"""
        
        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('group')
        
        self.path_realization.to_HDF(group)
        recalled_realization = self.path_realization.from_HDF(group)
        
        file.close()
        
        self.assertIsInstance(recalled_realization, RadarTargetRealization)
        assert_array_equal(self.target_position, recalled_realization.position)
        assert_array_equal(self.target_velocity, recalled_realization.velocity)
        self.assertEqual(self.cross_section, recalled_realization.cross_section)
        self.assertEqual(self.reflection_phase, recalled_realization.reflection_phase)


class TestInterferenceRealization(_TestRadarPathRealization[RadarInterferenceRealization]):
    """Test the radar interference realization class"""
    
    def _init_realization(self) -> RadarInterferenceRealization:
        return RadarInterferenceRealization(self.attenuate, self.static)

    def test_hdf_serialization(self) -> None:
        """Test serialization to and from HDF"""
        
        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('group')
        
        self.path_realization.to_HDF(group)
        recalled_realization = self.path_realization.from_HDF(group)
        
        file.close()
        
        self.assertIsInstance(recalled_realization, RadarInterferenceRealization)


class _TestRadarChannelRealization(Generic[RCRT], unittest.TestCase):
    """Test the radar channel realization base class"""
   
    def _init_realization(self) -> RCRT:
        ...
         
    def setUp(self) -> None:
        
        self.carrier_frequency = 1e9
        self.sampling_rate = 1e8
        
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (1, 1, 1)), pose=Transformation.From_Translation(np.array([0., 0., 0.], dtype=float)))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (3, 1, 1)), pose=Transformation.From_Translation(np.array([100., 0., 0.], dtype=float)))
        self.gain = 0.9876

        self.realization = self._init_realization()

    def test_properties(self) -> None:
        """Class properties should return initialization arguments"""
        
        self.assertIs(self.alpha_device, self.realization.alpha_device)
        self.assertIs(self.beta_device, self.realization.beta_device)
        self.assertEqual(self.gain, self.realization.gain)

    def test_propagate_state(self) -> None:
        """Propagation behaviour and channel state information should match"""

        signal = Signal(np.ones((self.alpha_device.antennas.num_transmit_antennas, 10)), self.sampling_rate, self.carrier_frequency)
        
        propagation = self.realization.propagate(signal)
        state = self.realization.state(self.alpha_device, self.beta_device, 0, self.sampling_rate, 20, 1 + propagation.signal.num_samples - signal.num_samples)
        state_propagation = state.propagate(signal)

        assert_array_almost_equal(propagation.signal.samples, state_propagation.samples[:, :propagation.signal.num_samples])

    def test_hdf_serialization(self) -> None:
        """Test serialization to and from HDF"""
        
        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('group')
        
        self.realization.to_HDF(group)
        recalled_realization = self.realization.From_HDF(group, self.alpha_device, self.beta_device)
        
        file.close()
        
        self.assertIsInstance(recalled_realization, RadarChannelRealization)
        self.assertIs(self.alpha_device, recalled_realization.alpha_device)
        self.assertIs(self.beta_device, recalled_realization.beta_device)
        self.assertEqual(self.gain, recalled_realization.gain)


class TestSingleTargetRadarChannelRealization(_TestRadarChannelRealization[SingleTargetRadarChannelRealization]):
    """Test single target radar channel realization"""
    
    def _init_realization(self) -> SingleTargetRadarChannelRealization:
        
        self.target_realization = RadarTargetRealization(np.ones(3), np.zeros(3), 1., 1)
        return SingleTargetRadarChannelRealization(self.alpha_device, self.beta_device, self.gain, self.target_realization)

    def test_null_hypothesis(self) -> None:
        """Null hypothesis realization should generate correct channel realization"""
        
        null_realization = self.realization.null_hypothesis()
        self.assertIsNone(null_realization.target_realization)

    def test_ground_truth(self) -> None:
        """Ground truth should return correct information"""
        
        assert_array_equal(np.array([self.target_realization.position]), self.realization.ground_truth())


class TestMultiTargetRadarChannelRealization(_TestRadarChannelRealization[MultiTargetRadarChannelRealization]):
    """Test the multi target radar channel realization class"""

    def _init_realization(self) -> MultiTargetRadarChannelRealization:
        
        self.target_realization = RadarTargetRealization(np.ones(3), np.zeros(3), 1., 1)
        self.interference_realization = RadarInterferenceRealization()
        return MultiTargetRadarChannelRealization(self.alpha_device, self.beta_device, self.gain, self.interference_realization, [self.target_realization,])

    def test_properties(self) -> None:
        
        super().test_properties()
        
        self.assertIs(self.interference_realization, self.realization.interference_realization)
        self.assertEqual(1, len(self.realization.target_realizations))
        self.assertIs(self.target_realization, self.realization.target_realizations[0])
        
    def test_null_hypothesis(self) -> None:
        """Null hypothesis realization should generate correct channel realization"""
        
        null_realization = self.realization.null_hypothesis()
        self.assertEqual(0, null_realization.num_targets)

    def test_ground_truth(self) -> None:
        """Ground truth should return correct information"""
        
        assert_array_equal(np.array([self.target_realization.position]), self.realization.ground_truth())


class _TestRadarChannelBase(Generic[RCT], unittest.TestCase):

    def _init_channel(self) -> RCT:
        ...  # pragma: no cover
    
    def setUp(self) -> None:
        
        self.random_generator = default_rng(42)
        self.random_root = Mock()
        self.random_root._rng = self.random_generator
        
        self.sampling_rate = 1e6
        self.carrier_frequency = 1e9
        
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (1, 1, 1)))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (1, 1, 1)))

        self.channel = self._init_channel()
        self.channel.random_mother = self.random_root

    def test_properties(self) -> None:
        """Class properties should return initialization arguments"""
        
        self.assertIs(self.alpha_device, self.channel.alpha_device)
        self.assertIs(self.beta_device, self.channel.beta_device)

    def test_attenuate_setget(self) -> None:
        """Attenuate property getter should return setter argument"""
        
        self.channel.attenuate = False
        self.assertFalse(self.channel.attenuate)
        
    def test_realize_target(self) -> None:
        """Test subroutine to realize a radar target"""
        
        cross_section = FixedCrossSection(1.)
        veolicty = np.zeros(3, dtype=float)
        pose = Transformation.From_Translation(np.array([100., 0., 0.], dtype=float))
        target = VirtualRadarTarget(cross_section, veolicty, pose)
        
        realization = self.channel._realize_target(target)
        
        self.assertAlmostEqual(2 * 100 / speed_of_light, realization.propagation_delay(self.alpha_device, self.beta_device), delta=1e-6)

    def test_realize_target_validation(self) -> None:
        """Target realization subroutine should raise errors on invalid parameter combinations"""
        
        cross_section = FixedCrossSection(1.)
        veolicty = np.zeros(3, dtype=float)
        pose = Transformation.From_Translation(np.array([100., 0., 0.], dtype=float))
        target = VirtualRadarTarget(cross_section, veolicty, pose)

            
        target.position = self.alpha_device.global_position
        with self.assertRaises(RuntimeError):
            _ = self.channel._realize_target(target)
            
        self.beta_device.position = np.array([1, 2, 3])
        target.position = self.beta_device.global_position
        with self.assertRaises(RuntimeError):
            _ = self.channel._realize_target(target)
            
    def test_null_hypothesis_validation(self) -> None:
        """Null hypothesis realization should raise RuntimeError on invalid internal state"""
        
        with self.assertRaises(RuntimeError):
            self.channel.null_hypothesis()
            
    def test_null_hypothesis(self) -> None:
        """The radar channel null hypothesis routine should create a valid null hypothesis"""
        
        signal = Signal(self.random_generator.normal(size=(self.alpha_device.antennas.num_transmit_antennas, 10)), self.alpha_device.sampling_rate, self.alpha_device.carrier_frequency)
        _ = self.channel.propagate(signal)
        
        null_hypothesis = self.channel.null_hypothesis()
        self.assertIsInstance(null_hypothesis, RadarChannelRealization)

    def test_yaml_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.alpha_device', new_callable=PropertyMock) as alpha_mock, \
             patch('hermespy.channel.Channel.beta_device', new_callable=PropertyMock) as beta_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            alpha_mock.return_value = None
            beta_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)
            
    def test_recall_realization(self) -> None:
        """Test recalling channel realizations from HDF"""
        
        realization = self.channel.realize()

        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('group')
        
        realization.to_HDF(group)
        recalled_realization = self.channel.recall_realization(group)
        
        file.close()
        
        self.assertIsInstance(recalled_realization, type(realization))
        self.assertIs(self.alpha_device, recalled_realization.alpha_device)
        self.assertIs(self.beta_device, recalled_realization.beta_device)


class TestSingleTargetRadarChannel(_TestRadarChannelBase[SingleTargetRadarChannel]):

    def _init_channel(self) -> SingleTargetRadarChannel:
        return SingleTargetRadarChannel(self.range, self.radar_cross_section,
                                        alpha_device=self.alpha_device,
                                        beta_device=self.beta_device)

    def setUp(self) -> None:

        self.range = 100.
        self.radar_cross_section = 1.
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
            self.channel.target_range = -1.
            
        with self.assertRaises(ValueError):
            self.channel.target_range = (1, 2, 3)
            
        with self.assertRaises(ValueError):
            self.channel.target_range = (3, 2)
            
        with self.assertRaises(ValueError):
            self.channel.target_range = (-1, 0)
            
        with self.assertRaises(ValueError):
            self.channel.target_range = 'wrong argument type'

    def test_radar_cross_section_get(self) -> None:
        """Radar cross section getter should return init param"""

        self.assertEqual(self.radar_cross_section, self.channel.radar_cross_section)

    def test_cross_section_validation(self) -> None:
        """Radar cross section property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.channel.radar_cross_section = -1.12345

        try:
            self.channel.radar_cross_section = 0.

        except ValueError:
            self.fail()

    def test_velocity_setget(self) -> None:
        """Velocity getter should return setter argument"""

        new_velocity = 20

        self.channel.target_velocity = new_velocity
        self.assertEqual(new_velocity, self.channel.target_velocity)

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
        expected_amplitude = ((speed_of_light / self.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        self.channel.target_range = expected_range

        propagation = self.channel.propagate(Signal(input_signal, self.sampling_rate, self.carrier_frequency))

        expected_output = np.hstack((np.zeros((1, delay_in_samples)), input_signal)) * expected_amplitude
        assert_array_almost_equal(abs(expected_output), np.abs(propagation.signal.samples[:, :expected_output.size]))

    def test_propagation_delay_noninteger_num_samples(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay falls in the
        middle of two sampling instants.
        """
        samples_per_symbol = 800
        num_pulses = 20
        delay_in_samples = 312

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = speed_of_light * (delay_in_samples + .5) / self.sampling_rate / 2
        expected_amplitude = ((speed_of_light / self.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        self.channel.target_range = expected_range

        propagation = self.channel.propagate(Signal(input_signal, self.sampling_rate, self.carrier_frequency))

        straddle_loss = np.sinc(.5)
        peaks = np.abs(propagation.signal.samples[:, delay_in_samples:input_signal.size:samples_per_symbol])

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
        expected_amplitude = ((speed_of_light / self.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

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

        propagation = self.channel.propagate(Signal(input_signal, self.sampling_rate, self.carrier_frequency))

        assert_array_almost_equal(np.abs(propagation.signal.samples[0, peaks_in_samples].flatten()), expected_straddle_amplitude)

    def test_doppler_shift(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given time variant delays on account of
        movement
        """

        velocity = 100
        self.channel.target_velocity = velocity

        num_samples = 100000
        sinewave_frequency = .25 * self.sampling_rate
        doppler_shift = 2 * velocity / speed_of_light * self.carrier_frequency

        time = np.arange(num_samples) / self.sampling_rate

        input_signal = np.sin(2 * np.pi * sinewave_frequency * time)
        propagation = self.channel.propagate(Signal(input_signal[np.newaxis, :], self.sampling_rate, self.carrier_frequency))

        input_freq = np.fft.fft(input_signal)
        output_freq = np.fft.fft(propagation.signal.samples.flatten()[-num_samples:])

        freq_resolution = self.sampling_rate / num_samples

        freq_in = np.argmax(np.abs(input_freq[:int(num_samples/2)])) * freq_resolution
        freq_out = np.argmax(np.abs(output_freq[:int(num_samples/2)])) * freq_resolution

        self.assertAlmostEqual(freq_out - freq_in, doppler_shift, delta=np.abs(doppler_shift)*.01)

    def test_no_echo(self) -> None:
        """Test if no echos are observed if target_exists flag is disabled"""
        
        samples_per_symbol = 500
        num_pulses = 15

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.target_exists = False
        propagation = self.channel.propagate(Signal(input_signal, self.sampling_rate))

        assert_array_almost_equal(propagation.signal.samples, np.zeros(propagation.signal.samples.shape))

    def test_no_attenuation(self) -> None:
        """Make sure the signal energy is preserved when the attenuate flag is disabled"""

        self.channel.attenuate = False
        self.channel.target_range = 10.

        input_signal = Signal(self._create_impulse_train(500, 15), self.sampling_rate)
        propagation = self.channel.propagate(input_signal)

        assert_array_almost_equal(input_signal.energy, propagation.signal.energy, 1)


class TestMultiTargetRadarChannel(_TestRadarChannelBase[MultiTargetRadarChannel]):
    """Test the multi target radar channel class"""
    
    def _init_channel(self) -> MultiTargetRadarChannel:
        return MultiTargetRadarChannel(alpha_device=self.alpha_device, beta_device=self.beta_device)
    
    def setUp(self) -> None:

        super().setUp()
        
        self.first_target = VirtualRadarTarget(FixedCrossSection(1.), velocity=np.array([10, 0, 0]), pose=Transformation.From_Translation(np.array([-10, 0, 0], dtype=float)))
        self.second_target = VirtualRadarTarget(FixedCrossSection(1.), velocity=np.array([-10, 0, 0]), pose=Transformation.From_Translation(np.array([10, 0, 0], dtype=float)))

        self.channel.add_target(self.first_target)
        self.channel.add_target(self.second_target)

    def test_add_virtual_target(self) -> None:
        """Test adding a new virtual radar target to the channel"""
        
        target = VirtualRadarTarget(FixedCrossSection(1.))
        self.channel.add_target(target)
        
        self.assertTrue(target in self.channel.targets)
        
    def test_add_physical_target(self) -> None:
        """Test adding a new physical radar target to the channel"""
        
        target = PhysicalRadarTarget(FixedCrossSection(1.), SimulatedDevice())
        self.channel.add_target(target)
        
        self.assertTrue(target in self.channel.targets)

    def test_make_target(self) -> None:
        """Test declaring a moveable as a radar target"""
        
        moveable = Moveable()
        crosse_section = FixedCrossSection(1.1234)
        new_target = self.channel.make_target(moveable, crosse_section)
        
        self.assertCountEqual([self.first_target, self.second_target, new_target], self.channel.targets)
            
    def test_realize_interference_monostatic(self) -> None:
        """Realization should not realize self-interference"""
        
        self.channel.beta_device = self.channel.alpha_device
        self.assertIsNone(self.channel._realize_interference())
            
    def test_realize_validation(self) -> None:
        """Realization should raise FloatingError if devices aren't specified"""
        
        with self.assertRaises(FloatingError):
            MultiTargetRadarChannel().realize()

    def test_realize(self) -> None:
        """Test SISO channel realization"""
        
        realization = self.channel.realize()
        self.assertEqual(2, len(realization.target_realizations))

    def test_null_hypothesis(self) -> None:
        """Test the null hypthesis realization routine"""
        
        _ = self.channel.realize()
        null_hypothesis = self.channel.null_hypothesis()
        
        self.assertEqual(0, null_hypothesis.num_targets)
        
    def test_null_hypothesis_static(self) -> None:
        """Test the null hypthoseis realization routine including a static target"""
        
        static_target = VirtualRadarTarget(FixedCrossSection(1.), pose=Transformation.From_Translation(np.array([10, 10, 10])), static=True)
        self.channel.add_target(static_target)
        
        _ = self.channel.realize()
        null_hypothesis = self.channel.null_hypothesis()
        
        self.assertEqual(1, null_hypothesis.num_targets)


del _TestRadarPathRealization
del _TestRadarChannelRealization
del _TestRadarChannelBase
