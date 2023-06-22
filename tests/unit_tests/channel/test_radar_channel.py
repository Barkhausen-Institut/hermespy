# -*- coding: utf-8 -*-
"""Test Radar Channel"""

from __future__ import annotations
import unittest
from unittest.mock import patch, Mock, PropertyMock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.constants import pi, speed_of_light

from hermespy.channel import SingleTargetRadarChannel, MultiTargetRadarChannel, VirtualRadarTarget, PhysicalRadarTarget, FixedCrossSection
from hermespy.channel.radar_channel import MultiTargetRadarChannelRealization, SingleTargetRadarChannelRealization, RadarChannelBase, RadarChannelRealization, RadarPathRealization, RadarInterferenceRealization, RadarTargetRealization
from hermespy.core import Direction, FloatingError, IdealAntenna, Moveable, Signal, Transformation, UniformArray
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization


__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarChannelRealizationMock(RadarChannelRealization):
    """Radar channel realization for base class testing"""
    
    def null_hypothesis(self, num_samples: int, sampling_rate: float) -> RadarChannelRealizationMock:

        impulse_response = self.ImpulseResponse([], self.gain, num_samples, sampling_rate, self.channel.transmitter, self.channel.receiver)
        return  RadarChannelRealizationMock(self.channel, self.gain, impulse_response)

    def ground_truth(self) -> np.ndarray:
        return np.empty((0, 3), dtype=np.float_)


class RadarChannelMock(RadarChannelBase[RadarChannelRealizationMock]):
    """Radar channel for base class testing"""
    
    def realize(self, num_samples: int, sampling_rate: float) -> RadarChannelRealizationMock:
      
        global_position = np.array([1, 1, 1], dtype=np.float_)  
        target_realization = RadarTargetRealization(0, 0, 0, 1, 2, np.eye(self.receiver.antennas.num_receive_antennas, self.transmitter.antennas.num_transmit_antennas), global_position, global_position)
        impulse_response = RadarChannelRealization.ImpulseResponse([target_realization], self.gain, num_samples, sampling_rate, self.transmitter, self.receiver)

        return RadarChannelRealizationMock(self, self.gain, impulse_response)


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


class TestRadarChannelRealization(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        
        self.transmitter = SimulatedDevice(carrier_frequency=1e9)
        self.receiver = SimulatedDevice(carrier_frequency=1e9)
        self.channel = RadarChannelMock(transmitter=self.transmitter, receiver=self.receiver)
        self.gain = 1.234
        self.impulse_response = self.gain**.5 * self.rng.normal(size=(1, 1, 1, 1))
        
        self.realization = RadarChannelRealizationMock(self.channel, self.gain, self.impulse_response)

    def test_init_validation(self) -> None:
        """Initializing should raise a ValueError o invalid arguments."""
        
        with self.assertRaises(ValueError):
            _ = RadarChannelRealizationMock(self.channel, -1, self.impulse_response)
        

class TestRadarChannelBase(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        self.carrier_frequency = 1e9
        
        self.transmitter = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (2, 1, 1)))
        self.receiver = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna, .01, (3, 1, 1)))
        self.channel = RadarChannelMock(self, transmitter=self.transmitter, receiver=self.receiver)
        
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
        
        realization = self.channel._realize_target(1e9, target)
        
        self.assertAlmostEqual(2 * 100 / speed_of_light, realization.delay)
        self.assertSequenceEqual((self.receiver.antennas.num_receive_antennas, self.transmitter.antennas.num_transmit_antennas), realization.mimo_response.shape)

    def test_realize_target_validation(self) -> None:
        """Target realization subroutine should raise errors on invalid parameter combinations"""
        
        cross_section = FixedCrossSection(1.)
        veolicty = np.zeros(3, dtype=float)
        pose = Transformation.From_Translation(np.array([100., 0., 0.], dtype=float))
        target = VirtualRadarTarget(cross_section, veolicty, pose)
        
        with self.assertRaises(ValueError):
            _ = self.channel._realize_target(0, target)
            
        with self.assertRaises(ValueError):
            _ = self.channel._realize_target(-1, target)
            
        target.position = self.transmitter.global_position
        with self.assertRaises(RuntimeError):
            _ = self.channel._realize_target(1, target)
            
        self.receiver.position = np.array([1, 2, 3])
        target.position = self.receiver.global_position
        with self.assertRaises(RuntimeError):
            _ = self.channel._realize_target(1, target)
            
        self.channel.transmitter = None
        with self.assertRaises(FloatingError):
            _ = self.channel._realize_target(1, target)
            
    def test_null_hypothesis_validation(self) -> None:
        """Null hypothesis realization should raise RuntimeError on invalid internal state"""
        
        with self.assertRaises(RuntimeError):
            self.channel.null_hypothesis(1, 1)
            
    def test_null_hypothesis(self) -> None:
        """The radar channel null hypothesis routine should create a valid null hypothesis"""
        
        signal = Signal(self.rng.normal(size=(self.transmitter.antennas.num_transmit_antennas, 10)), self.transmitter.sampling_rate, self.transmitter.carrier_frequency)
        _ = self.channel.propagate(signal)
        
        null_hypothesis = self.channel.null_hypothesis(10, self.transmitter.sampling_rate)
        self.assertEqual(self.transmitter.antennas.num_transmit_antennas, null_hypothesis.num_transmit_streams)
        self.assertEqual(self.receiver.antennas.num_receive_antennas, null_hypothesis.num_receive_streams)
        self.assertEqual(10, null_hypothesis.num_samples)


class TestRadarPathRealization(unittest.TestCase):
    """Test the radar path realization class"""
    
    def setUp(self) -> None:
        
        self.phase_shift = 1.
        self.delay = 2.
        self.doppler_shift = 3.
        self.doppler_velocity = 312.
        self.power_factor = 4.
        self.mimo_response = np.array([[[1, 2], [3, 4]]])
        self.global_position = np.array([1, 2, 3])
        self.global_velocity = np.array([0, 0, 312])
        self.static = True
        self.realization = RadarPathRealization(self.phase_shift, self.delay, self.doppler_shift, self.doppler_velocity, self.power_factor, self.mimo_response, self.global_position, self.global_velocity, self.static)
        
    def test_properties(self) -> None:
        """Class properties should return initialization arguments"""
        
        self.assertEqual(self.phase_shift, self.realization.phase_shift)
        self.assertEqual(self.delay, self.realization.delay)
        self.assertEqual(self.doppler_shift, self.realization.doppler_shift)
        self.assertEqual(self.doppler_velocity, self.realization.doppler_velocity)
        self.assertEqual(self.power_factor, self.realization.power_factor)
        assert_array_equal(self.mimo_response, self.realization.mimo_response)
        assert_array_equal(self.global_position, self.realization.global_position)
        assert_array_equal(self.global_velocity, self.realization.global_velocity)
        self.assertEqual(self.static, self.realization.static)


class TestSingleTargetRadarChannelRealization(unittest.TestCase):
    """Test the single target radar channel realization class"""
    
    def setUp(self) -> None:
        
        self.device = SimulatedDevice(carrier_frequency=1e9, antennas=UniformArray(IdealAntenna, .01, (2, 1, 1)))
        self.channel = SingleTargetRadarChannel(1., 1.)
        self.channel.transmitter = self.device
        self.channel.receiver = self.device
        
        self.gain = 2.
        self.target_realization = RadarTargetRealization(1., 2., 3., 4., 5., np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]), np.array([0, 0, 4.]))
        self.num_samples = 10
        self.sampling_rate = 1.234
        
        self.realization = SingleTargetRadarChannelRealization(self.channel, self.gain, self.target_realization, self.num_samples, self.sampling_rate)

    def test_properties(self) -> None:
        """Class properties should return initialization arguments"""
        
        self.assertEqual(self.gain, self.realization.gain)
        self.assertIs(self.target_realization, self.realization.target_realization)

    def test_null_hypothesis(self) -> None:
        """Null hypothesis realization should generate correct channel realization"""
        
        null_realization = self.realization.null_hypothesis(10, 1e8)
        
        self.assertIsInstance(null_realization, SingleTargetRadarChannelRealization)

    def test_ground_trutch(self) -> None:
        """Ground truth should return correct information"""
        
        assert_array_equal(np.array([[1, 2, 3]]), self.realization.ground_truth())


class TestMultiTargetRadarChannelRealization(unittest.TestCase):
    """Test the multi target radar channel realization class"""
    
    def setUp(self) -> None:
    
        self.device = SimulatedDevice(carrier_frequency=1e9, antennas=UniformArray(IdealAntenna, .01, (2, 1, 1)))
        self.channel = SingleTargetRadarChannel(1., 1.)
        self.channel.transmitter = self.device
        self.channel.receiver = self.device
        
        self.gain = 2.
        self.target_realization = RadarTargetRealization(1., 2., 3., 4., 5., np.array([[1, 2], [3, 4]]), np.array([0, 1, 2]), np.array([1, 2, 3]))
        self.interference_realization = RadarInterferenceRealization(1., 2., 3., 4., 5., np.array([[2, 5], [1, 3]]), np.array([0, 1, 2]), np.array([4, 5, 6]))
        self.num_samples = 10
        self.sampling_rate = 1.234
        
        self.realization = MultiTargetRadarChannelRealization(self.channel, self.gain, self.interference_realization, [self.target_realization], self.num_samples, self.sampling_rate)

    def test_properties(self) -> None:
        """Class properties should return initialization arguments"""
        
        self.assertEqual(self.gain, self.realization.gain)
        self.assertIs(self.interference_realization, self.realization.interference_realization)
        self.assertSequenceEqual([self.target_realization], self.realization.target_realizations)

    def test_null_hypothesis(self) -> None:
        """Null hypothesis realization should generate correct channel realization"""
        
        null_realization = self.realization.null_hypothesis(self.num_samples, self.sampling_rate)
        
        self.assertIsInstance(null_realization, MultiTargetRadarChannelRealization)
        
    def test_ground_trutch(self) -> None:
        """Ground truth should return correct information"""
        
        assert_array_equal(np.array([[0, 1, 2]]), self.realization.ground_truth())


class TestMultiTargetRadarChannel(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.carrier_frequency = 1e9
        self.transmitter = SimulatedDevice(carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([0, 0, 0], dtype=float)))
        self.receiver = SimulatedDevice(carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([20, 0, 0], dtype=float)))
        
        self.first_target = VirtualRadarTarget(FixedCrossSection(1.), velocity=np.array([10, 0, 0]), pose=Transformation.From_Translation(np.array([-10, 0, 0], dtype=float)))
        self.second_target = VirtualRadarTarget(FixedCrossSection(1.), velocity=np.array([-10, 0, 0]), pose=Transformation.From_Translation(np.array([10, 0, 0], dtype=float)))
        self.channel = MultiTargetRadarChannel(transmitter=self.transmitter, receiver=self.receiver)
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
        
    def test_realize_interference_validation(self) -> None:
        """Interference realization subroutine should raise errors on invalid parameters and states"""
        
        with self.assertRaises(ValueError):
            self.channel._realize_interference(0.)
            
        with self.assertRaises(FloatingError):
            MultiTargetRadarChannel()._realize_interference(1.2345)
            
        self.transmitter.pose.translation = np.zeros(3)
        self.receiver.pose.translation = np.zeros(3)
        
        with self.assertRaises(RuntimeError):
            self.channel._realize_interference(1.234)
            
    def test_realize_interference_monostatic(self) -> None:
        """Realization should not realize self-interference"""
        
        self.channel.receiver = self.channel.transmitter
        self.assertIsNone(self.channel._realize_interference(1.234))
            
    def test_realize_validation(self) -> None:
        """Realization should raise FloatingError if devices aren't specified"""
        
        with self.assertRaises(FloatingError):
            MultiTargetRadarChannel().realize(1, 1.)

    def test_siso_realize(self) -> None:
        """Test SISO channel realization"""
        
        realization = self.channel.realize(200, 1e8)
        
        self.assertEqual(2, len(realization.target_realizations))
        
        self.assertEqual(1, realization.num_receive_streams)
        self.assertEqual(1, realization.num_transmit_streams)
        self.assertEqual(200, realization.num_samples)
        
    def test_mimo_realize(self) -> None:
        """Test MIMO channel realization"""
        
        antenna_spacing = .5 * self.transmitter.wavelength
        self.transmitter.antennas = UniformArray(IdealAntenna, antenna_spacing, (2, 2, 1))
        self.receiver.antennas = UniformArray(IdealAntenna, antenna_spacing, (2, 2, 1))
        
        realization = self.channel.realize(200, 1e8)
        
        self.assertEqual(2, len(realization.target_realizations))
        
        self.assertEqual(4, realization.num_receive_streams)
        self.assertEqual(4, realization.num_transmit_streams)
        self.assertEqual(200, realization.num_samples)

    def test_null_hypothesis(self) -> None:
        """Test the null hypthesis realization routine"""
        
        num_samples = 200
        sampling_rate = 1e8
        one_hypothesis = self.channel.realize(num_samples, sampling_rate)
        null_hypothesis = self.channel.null_hypothesis(num_samples, sampling_rate, one_hypothesis)
        
        self.assertEqual(1, null_hypothesis.num_receive_streams)
        self.assertEqual(1, null_hypothesis.num_transmit_streams)
        self.assertEqual(200, null_hypothesis.num_samples)
        self.assertEqual(0, len(null_hypothesis.target_realizations))
        self.assertAlmostEqual(0., float(np.linalg.norm(null_hypothesis.state)))
        
    def test_null_hypothesis_static(self) -> None:
        """Test the null hypthoseis realization routine including a static target"""
        
        static_target = VirtualRadarTarget(FixedCrossSection(1.), pose=Transformation.From_Translation(np.array([10, 10, 10])), static=True)
        self.channel.add_target(static_target)
        
        num_samples = 200
        sampling_rate = 1e8
        one_hypothesis = self.channel.realize(num_samples, sampling_rate)
        null_hypothesis = self.channel.null_hypothesis(num_samples, sampling_rate, one_hypothesis)
        
        self.assertEqual(1, null_hypothesis.num_receive_streams)
        self.assertEqual(1, null_hypothesis.num_transmit_streams)
        self.assertEqual(num_samples, null_hypothesis.num_samples)
        self.assertEqual(1, len(null_hypothesis.target_realizations))
        self.assertLess(0., float(np.linalg.norm(null_hypothesis.state)))
        
    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.transmitter', new_callable=PropertyMock) as transmitter_mock, \
             patch('hermespy.channel.Channel.receiver', new_callable=PropertyMock) as receiver_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)


class TestSingleTargetRadarChannel(unittest.TestCase):

    def setUp(self) -> None:

        self.range = 100.
        self.radar_cross_section = 1.

        self.random_generator = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.transmitter = SimulatedDevice(carrier_frequency=1e9, sampling_rate=1e6)
        self.receiver = self.transmitter

        self.target_exists = True
        self.channel = SingleTargetRadarChannel(self.range, self.radar_cross_section,
                                                transmitter=self.transmitter,
                                                receiver=self.receiver,
                                                target_exists=self.target_exists)
        self.channel.random_mother = self.random_node

        self.expected_delay = 2 * self.range / speed_of_light

    def test_init(self) -> None:
        """The object initialization should properly store all parameters"""

        self.assertEqual(self.range, self.channel.target_range)
        self.assertIs(self.radar_cross_section, self.channel.radar_cross_section)
        self.assertIs(self.transmitter, self.channel.transmitter)
        self.assertIs(self.receiver, self.channel.receiver)
        self.assertIs(self.target_exists, self.channel.target_exists)

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

    def test_target_exists_setget(self) -> None:
        """Target exists flag getter should return setter argument"""

        new_target_exists = False
        self.channel.target_exists = new_target_exists
        self.assertEqual(new_target_exists, self.channel.target_exists)

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

    def test_realize_anchored_validation(self) -> None:
        """Impulse response should raise FloatingError if not anchored to a device"""

        with patch.object(RadarChannelBase, 'transmitter', None), self.assertRaises(FloatingError):
            _ = self.channel.realize(0, 1.)

    def test_realize_carrier_frequency_validation(self) -> None:
        """Impulse response should raise RuntimeError if device carrier frequencies are smaller or equal to zero"""

        self.transmitter.carrier_frequency = 0.

        with self.assertRaises(ValueError):
            _ = self.channel.realize(0, 1.)

    def test_realize_interference_validation(self) -> None:
        """Impulse response should raise RuntimeError if not configured as a self-interference channel"""

        with patch.object(SingleTargetRadarChannel, 'receiver', None), self.assertRaises(RuntimeError):
            _ = self.channel.realize(0, 1.)

    def _create_impulse_train(self, interval_in_samples: int, number_of_pulses: int):

        interval = interval_in_samples / self.transmitter.sampling_rate

        number_of_samples = int(np.ceil(interval * self.transmitter.sampling_rate * number_of_pulses))
        output_signal = np.zeros((1, number_of_samples), dtype=complex)

        interval_in_samples = int(np.around(interval * self.transmitter.sampling_rate))

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

        expected_range = speed_of_light * delay_in_samples / self.transmitter.sampling_rate / 2
        expected_amplitude = ((speed_of_light / self.transmitter.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        self.channel.target_range = expected_range

        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        expected_output = np.hstack((np.zeros((1, delay_in_samples)), input_signal)) * expected_amplitude
        assert_array_almost_equal(abs(expected_output), np.abs(output[0].samples[:, :expected_output.size]))

    def test_propagation_delay_noninteger_num_samples(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay falls in the
        middle of two sampling instants.
        """
        samples_per_symbol = 800
        num_pulses = 20
        delay_in_samples = 312

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = speed_of_light * (delay_in_samples + .5) / self.transmitter.sampling_rate / 2
        expected_amplitude = ((speed_of_light / self.transmitter.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        self.channel.target_range = expected_range

        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        straddle_loss = np.sinc(.5)
        peaks = np.abs(output[0].samples[:, delay_in_samples:input_signal.size:samples_per_symbol])

        assert_array_almost_equal(peaks, expected_amplitude * straddle_loss * np.ones(peaks.shape))

    def test_propagation_delay_doppler(self) -> None:
        """
        Test if the received signal corresponds to a frequency-shifted version of the transmitted signal with the
        expected Doppler shift
        """

        samples_per_symbol = 50
        num_pulses = 100
        initial_delay_in_samples = 1000
        expected_range = speed_of_light * initial_delay_in_samples / self.transmitter.sampling_rate / 2
        velocity = 10
        expected_amplitude = ((speed_of_light / self.transmitter.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        initial_delay = initial_delay_in_samples / self.transmitter.sampling_rate

        timestamps_impulses = np.arange(num_pulses) * samples_per_symbol / self.transmitter.sampling_rate
        traveled_distances = velocity * timestamps_impulses
        delays = initial_delay + 2 * traveled_distances / speed_of_light
        expected_peaks = timestamps_impulses + delays
        peaks_in_samples = np.around(expected_peaks * self.transmitter.sampling_rate).astype(int)
        straddle_delay = expected_peaks - peaks_in_samples / self.transmitter.sampling_rate
        relative_straddle_delay = straddle_delay * self.transmitter.sampling_rate
        expected_straddle_amplitude = np.sinc(relative_straddle_delay) * expected_amplitude

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.target_range = expected_range
        self.channel.velocity = velocity

        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        assert_array_almost_equal(np.abs(output[0].samples[0, peaks_in_samples].flatten()), expected_straddle_amplitude)

    def test_doppler_shift(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given time variant delays on account of
        movement
        """

        velocity = 100
        self.channel.target_velocity = velocity

        num_samples = 100000
        sinewave_frequency = .25 * self.transmitter.sampling_rate
        doppler_shift = 2 * velocity / speed_of_light * self.transmitter.carrier_frequency

        time = np.arange(num_samples) / self.transmitter.sampling_rate

        input_signal = np.sin(2 * np.pi * sinewave_frequency * time)
        output, _, _ = self.channel.propagate(Signal(input_signal[np.newaxis, :], self.transmitter.sampling_rate))

        input_freq = np.fft.fft(input_signal)
        output_freq = np.fft.fft(output[0].samples.flatten()[-num_samples:])

        freq_resolution = self.transmitter.sampling_rate / num_samples

        freq_in = np.argmax(np.abs(input_freq[:int(num_samples/2)])) * freq_resolution
        freq_out = np.argmax(np.abs(output_freq[:int(num_samples/2)])) * freq_resolution

        self.assertAlmostEqual(freq_out - freq_in, doppler_shift, delta=np.abs(doppler_shift)*.01)

    def test_no_echo(self) -> None:
        """Test if no echos are observed if target_exists flag is disabled"""
        
        samples_per_symbol = 500
        num_pulses = 15

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.target_exists = False
        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        assert_array_almost_equal(output[0].samples, np.zeros(output[0].samples.shape))

    def test_no_attenuation(self) -> None:
        """Make sure the signal energy is preserved when the attenuate flag is disabled"""

        self.channel.attenuate = False
        self.channel.target_range = 10.

        input_signal = Signal(self._create_impulse_train(500, 15), self.transmitter.sampling_rate)
        output, _, _ = self.channel.propagate(input_signal)

        assert_array_almost_equal(input_signal.energy, output[0].energy, 1)

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.transmitter', new_callable=PropertyMock) as transmitter_mock, \
             patch('hermespy.channel.Channel.receiver', new_callable=PropertyMock) as receiver_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)
