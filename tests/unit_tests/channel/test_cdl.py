# -*- coding: utf-8 -*-

from abc import abstractmethod
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal
from h5py import File
from scipy.constants import speed_of_light

from hermespy.core import DenseSignal, Transformation
from hermespy.simulation import DeviceState, SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.simulation.animation import StaticTrajectory, TrajectorySample
from hermespy.channel.channel import LinkState
from hermespy.channel.cdl import CDL, CDLType, LOSState, O2IState, FactoryType, IndoorFactory, IndoorOffice, OfficeType, RuralMacrocells, UrbanMacrocells, UrbanMicrocells
from hermespy.channel.cdl.cluster_delay_lines import ClusterDelayLineBase, ClusterDelayLineRealization, ClusterDelayLineSample, ClusterDelayLineSampleParameters, DelayNormalization
from unit_tests.utils import assert_signals_equal, SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestClusterDelayLineSample(TestCase):
    """Test the sample of the 3GPP Cluster Delay Line Model"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
    
        self.carrier_frequency = 1e9
        self.bandwidth = 1e6
        self.line_of_sight = False
        self.rice_factor = 0.9
        self.azimuth_of_arrival = self.rng.normal(size=(6, 20))
        self.zenith_of_arrival = self.rng.normal(size=(6, 20))
        self.azimuth_of_departure = self.rng.normal(size=(6, 20))
        self.zenith_of_departure = self.rng.normal(size=(6, 20))
        self.delay_offset = 1e-8
        self.cluster_delays = np.arange(6) * 1e-6
        self.cluster_delay_spread = 2e-6
        self.cluster_powers = self.rng.rayleigh(size=6)
        self.polarization_transformations = self.rng.normal(size=(2, 2, 6, 20)) + 1j * self.rng.normal(size=(2, 2, 6, 20))
        
        self.transmitter_state = DeviceState(
            TrajectorySample(
                0.0,
                Transformation.From_Translation(np.array([2, 3, 4])),
                np.array([1, 2, 3]),
            ),
            self.carrier_frequency,
            self.bandwidth,
            SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)).state(Transformation.From_Translation(np.array([2, 3, 4]))),
            Mock(),
        )
        self.receiver_state = DeviceState(
            TrajectorySample(
                0.0,
                Transformation.From_Translation(np.array([5, 6, 7])),
                np.array([4, 5, 6]),
            ),
            self.carrier_frequency,
            self.bandwidth,
            SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)).state(Transformation.From_Translation(np.array([2, 3, 4]))),
            Mock(),
        )
        
        self.sample = ClusterDelayLineSample(
            self.line_of_sight,
            self.rice_factor,
            self.azimuth_of_arrival,
            self.zenith_of_arrival,
            self.azimuth_of_departure,
            self.zenith_of_departure,
            self.delay_offset,
            self.cluster_delays,
            self.cluster_delay_spread,
            self.cluster_powers,
            self.polarization_transformations,
            LinkState(self.transmitter_state, self.receiver_state, self.carrier_frequency, self.bandwidth, 0.0)
        )
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertIs(self.transmitter_state, self.sample.transmitter_state)
        self.assertIs(self.receiver_state, self.sample.receiver_state)
        self.assertEqual(self.carrier_frequency, self.sample.carrier_frequency)
        self.assertEqual(self.bandwidth, self.sample.bandwidth)
        self.assertEqual(self.line_of_sight, self.sample.line_of_sight)
        self.assertEqual(self.rice_factor, self.sample.rice_factor)
        assert_array_equal(self.azimuth_of_arrival, self.sample.azimuth_of_arrival)
        assert_array_equal(self.zenith_of_arrival, self.sample.zenith_of_arrival)
        assert_array_equal(self.azimuth_of_departure, self.sample.azimuth_of_departure)
        assert_array_equal(self.zenith_of_departure, self.sample.zenith_of_departure)
        self.assertEqual(self.delay_offset, self.sample.delay_offset)
        assert_array_equal(self.cluster_delays, self.sample.cluster_delays)
        self.assertEqual(self.cluster_delay_spread, self.sample.cluster_delay_spread)
        assert_array_equal(self.cluster_powers, self.sample.cluster_powers)
        assert_array_equal(self.polarization_transformations, self.sample.polarization_transformations)

    def test_propagate_state(self) -> None:
        """Generated channel state should correctly predict signal propagation"""

        signal_samples = self.rng.standard_normal((self.sample.num_transmit_antennas, 100)) + 1j * self.rng.standard_normal((self.sample.num_transmit_antennas, 100))
        signal = DenseSignal(signal_samples, self.bandwidth, self.carrier_frequency)

        signal_propagation = self.sample.propagate(signal)
        state_propagation = self.sample.state(signal.num_samples, 1 + signal_propagation.num_samples - signal.num_samples).propagate(signal)

        assert_signals_equal(self, signal_propagation, state_propagation)

    def test_plot_power_delay(self) -> None:
        """Test the power delay profile visualization routine"""

        with patch("matplotlib.pyplot.figure") as mock_figure:
            self.sample.plot_power_delay()
            mock_figure.assert_called_once()
            
    def test_plot_angles(self) -> None:
        """Test the angle visualization routine"""

        with patch("matplotlib.pyplot.figure") as mock_figure:
            self.sample.plot_angles()
            mock_figure.assert_called_once()

    def test_plot_rays(self) -> None:
        """Test the ray visualization routine"""

        with patch("matplotlib.pyplot.figure") as mock_figure:
            self.sample.plot_rays()
            mock_figure.assert_called_once()

    def test_angular_spread_validation(self) -> None:
        """Angular spread routine should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.sample._angular_spread(np.arange(3), np.arange(4))

    def test_azimuth_arrival_spread(self) -> None:
        """Test the azimuth of arrival spread calculation"""

        spread = self.sample.azimuth_arrival_spread

    def test_azimuth_departure_spread(self) -> None:
        """Test the azimuth of departure spread calculation"""

        spread = self.sample.azimuth_departure_spread

    def test_zenith_arrival_spread(self) -> None:
        """Test the zenith of arrival spread calculation"""

        spread = self.sample.zenith_arrival_spread

    def test_zenith_departure_spread(self) -> None:
        """Test the zenith of departure spread calculation"""

        spread = self.sample.zenith_departure_spread
        
    def test_reciprocal(self) -> None:
        """Reciprocal channel should simply switch transmitter and dreceiver states"""
        
        reciprocal_sample = self.sample.reciprocal(LinkState(
            self.receiver_state, self.transmitter_state, self.carrier_frequency, self.bandwidth, 0.0)
        )
        
        self.assertIs(self.transmitter_state, reciprocal_sample.receiver_state)
        self.assertIs(self.receiver_state, reciprocal_sample.transmitter_state)

    def test_angle_visualization(self) -> None:
        """Test the angle visualization routine"""
        
        with SimulationTestContext():
            self.sample.plot_angles()


class TestClusterDelayLineSampleParameters(TestCase):
    """Test the 3GPP CDL sample parameter dataclass"""
    
    def setUp(self) -> None:
        
        self.carrier_frequency = 1e9
        self.distance_3d = 100
        self.distance_2d = 80
        self.base_height = 1.5
        self.transmitter_height = 1.5
        
        self.parameters = ClusterDelayLineSampleParameters(
            self.carrier_frequency,
            self.distance_3d,
            self.distance_2d,
            self.base_height,
            self.transmitter_height,
        )
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertEqual(self.carrier_frequency, self.parameters.carrier_frequency)
        self.assertEqual(self.distance_3d, self.parameters.distance_3d)
        self.assertEqual(self.distance_2d, self.parameters.distance_2d)
        self.assertEqual(self.base_height, self.parameters.base_height)
        self.assertEqual(self.transmitter_height, self.parameters.terminal_height)


class TestClusterDelayLine(TestCase):
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.carrier_frequency = 1e9
        self.bandwidth = 1e8
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.bandwidth, pose=Transformation.From_Translation(np.array([0, 0, 0])))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.bandwidth, pose=Transformation.From_Translation(np.array([100, 100, 0])))

        self.gain = 0.98
        self.delay_normalization = DelayNormalization.TOF
        self.oxygen_absorption = False
        
        self.model = self._init_model()
        self.model._rng = self.rng
        
    @abstractmethod
    def _init_model(self) -> ClusterDelayLineBase:
        """Initialize the model under test"""
        ... 
    
    @abstractmethod
    def _large_scale_states(self) -> list:
        """Large scale options for the model"""
        ... 
    
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertEqual(self.gain, self.model.gain)
        self.assertEqual(self.delay_normalization, self.model.delay_normalization)
        self.assertEqual(self.oxygen_absorption, self.model.oxygen_absorption)
        
    def _test_propagation(self, sample: ClusterDelayLineSample) -> None:
        
        num_samples = 100
        transmitted_signal = DenseSignal(self.rng.normal(size=(sample.num_transmit_antennas, num_samples)) + 1j * self.rng.normal(size=(sample.num_transmit_antennas, num_samples)), self.bandwidth, self.carrier_frequency)
        initial_power = transmitted_signal.power
        
        propagated_signal = sample.propagate(transmitted_signal)
        
        self.assertEqual(sample.num_receive_antennas, propagated_signal.num_streams)
        self.assertEqual(sample.bandwidth, propagated_signal.sampling_rate)
        self.assertEqual(sample.carrier_frequency, propagated_signal.carrier_frequency)
        
        # ToDo: Test power conservation
        
    def test_sample_validation(self) -> None:
        """Sample routine should raise ValueError if devices are colocated"""
        
        self.alpha_device.trajectory = StaticTrajectory(Transformation.From_Translation(np.zeros(3)))
        self.beta_device.trajectory = StaticTrajectory(Transformation.From_Translation(np.zeros(3)))

        with self.assertRaises(RuntimeError):
            self.model.realize().sample(self.alpha_device, self.beta_device)

    def test_random_realization(self) -> None:
        """Test a random large scale model realization"""
        
        realization: ClusterDelayLineRealization = self.model.realize()
        sample: ClusterDelayLineSample = realization.sample(self.alpha_device, self.beta_device, self.carrier_frequency, self.bandwidth)
        self._test_propagation(sample)

    def test_expected_realization(self) -> None:
        """Test a deterministic large scale model realization"""
        
        for expected_state in self._large_scale_states():
    
            self.model.expected_state = expected_state
            realization: ClusterDelayLineRealization = self.model.realize()            
            self.assertIs(expected_state, realization.expected_state)
            
            sample: ClusterDelayLineSample = realization.sample(self.alpha_device, self.beta_device)
            self._test_propagation(sample)
            
    def test_expected_scale(self) -> None:
        """Test the expected amplitude scaling"""
        
        unit_energy_signal = DenseSignal(np.ones((self.alpha_device.num_transmit_antennas, 100)) / 10, self.bandwidth, self.carrier_frequency)
        num_attempts = 100
        
        cumulated_propagated_energy = np.zeros((self.beta_device.num_receive_antennas), dtype=np.float_)
        cumulated_expected_scale = 0.0
        for _ in range(num_attempts):
            realization = self.model.realize()
            sample = realization.sample(self.alpha_device, self.beta_device)
            propagated_signal = sample.propagate(unit_energy_signal)
            cumulated_propagated_energy += propagated_signal.energy
            cumulated_expected_scale += sample.expected_energy_scale
            
        mean_propagated_energy = cumulated_propagated_energy / num_attempts
        mean_expected_energy = (cumulated_expected_scale / num_attempts) ** 2
        
        self.assertAlmostEqual(mean_propagated_energy, mean_expected_energy, delta=1e-1)
    
    def test_propagation_time_of_flight(self) -> None:
        """Test time of flight delay simulation"""
        
        self.model.delay_normalization = DelayNormalization.TOF
        realization: ClusterDelayLineRealization = self.model.realize()
        sample: ClusterDelayLineSample = realization.sample(self.alpha_device, self.beta_device)
        
        signal = DenseSignal(np.ones((sample.num_transmit_antennas, 1)), self.bandwidth, self.carrier_frequency)
        propagated_signal = sample.propagate(signal)
        
        expected_tof_delay = np.linalg.norm(self.alpha_device.state(0).position - self.beta_device.state(0).position) / speed_of_light * self.bandwidth
        delay = np.argmax(np.abs(propagated_signal[0, :]) != 0.0)
        
        self.assertAlmostEqual(expected_tof_delay, delay, delta=1)
        
    def test_propagation_excess_delay(self) -> None:
        """Test excess delay simulation"""
        
        self.model.delay_normalization = DelayNormalization.ZERO
        realization: ClusterDelayLineRealization = self.model.realize()
        sample: ClusterDelayLineSample = realization.sample(self.alpha_device, self.beta_device)
        
        signal = DenseSignal(np.ones((sample.num_transmit_antennas, 1)), self.bandwidth, self.carrier_frequency)
        propagated_signal = sample.propagate(signal)
        
        delay = np.argmax(np.abs(propagated_signal[0, :]) != 0.0)
        
        self.assertAlmostEqual(0, delay, delta=1)
        
    def test_oxygen_absorption(self) -> None:
        """Test oxygen power absorption modeling"""
        
        signal = DenseSignal(np.ones((self.alpha_device.num_transmit_antennas, 1)), self.bandwidth, self.carrier_frequency)

        ideal_propagation = self.model.realize().sample(self.alpha_device, self.beta_device).propagate(signal)
        
        self.model.oxygen_absorption = True
        absorbed_propagation = self.model.realize().sample(self.alpha_device, self.beta_device).propagate(signal)
        
        self.assertGreater(np.sum(ideal_propagation.power), np.sum(absorbed_propagation.power))
            
    def test_reciprocal_sample(self) -> None:
        """Reciprocal sample should simply switch transmitter and receiver states"""
        
        realization: ClusterDelayLineRealization = self.model.realize()
        sample: ClusterDelayLineSample = realization.sample(self.alpha_device, self.beta_device)
        
        reciprocal_sample = realization.reciprocal_sample(sample, self.alpha_device, self.beta_device)
        reciprocal_sample = realization.reciprocal_sample(sample, self.beta_device, self.alpha_device)
            
        # Alter state
        self.alpha_device.carrier_frequency = 2e9
        reciprocal_sample = realization.reciprocal_sample(sample, self.alpha_device, self.beta_device)
            
    def test_hdf_serialization(self) -> None:
        """Test the realization serialization to and from HDF"""
        
        expected_realization: ClusterDelayLineRealization = self.model.realize()
        
        file = File("test.hdf", "w", "core")
        group = file.create_group("realization")
        
        expected_realization.to_HDF(group)
        recalled_realization: ClusterDelayLineRealization = self.model.recall_realization(group)
        
        file.close()
        
        self.assertEqual(expected_realization.expected_state, recalled_realization.expected_state)
        self.assertEqual(expected_realization.gain, recalled_realization.gain)


class TestIndoorFactory(TestClusterDelayLine):
    
    def _init_model(self) -> IndoorFactory:
        return IndoorFactory(2000, 3000, FactoryType.HH, 1.0, self.gain, delay_normalization=self.delay_normalization, oxygen_absorption=self.oxygen_absorption)
    
    def _large_scale_states(self) -> list:
        return list(LOSState)
    
    def test_volume_setget(self) -> None:
        """Volume getter should return setter argument"""
        
        self.model.volume = 1000
        self.assertEqual(1000, self.model.volume)
        
    def test_volume_validation(self) -> None:
        """Volume should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.model.volume = -1000
            
        with self.assertRaises(ValueError):
            self.model.volume = 0
            
    def test_surface_setget(self) -> None:
        """Surface getter should return setter argument"""
        
        self.model.surface = 1000
        self.assertEqual(1000, self.model.surface)
        
    def test_surface_validation(self) -> None:
        """Surface should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.model.surface = -1000
            
        with self.assertRaises(ValueError):
            self.model.surface = 0
            
    def test_clutter_height_setget(self) -> None:
        """Clutter height getter should return setter argument"""
        
        self.model.clutter_height = 5
        self.assertEqual(5, self.model.clutter_height)
        
    def test_clutter_height_validation(self) -> None:
        """Clutter height should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.model.clutter_height = -1000
            
        with self.assertRaises(ValueError):
            self.model.clutter_height = 20
    
    def test_factory_type(self) -> None:
        """Test different factory types"""
        
        for factory_type in list(FactoryType):
            self.model.factory_type = factory_type
            realization = self.model.realize()
            sample = realization.sample(self.alpha_device, self.beta_device, self.carrier_frequency, self.bandwidth)
            self._test_propagation(sample)

    def test_hdf_serialization_expected_state(self) -> None:
        """Test HDF serialization with a fixed expected state"""
    
        expected_state = LOSState.LOS
        self.model.expected_state = expected_state
        
        realization = self.model.realize()
        
        file = File("test.hdf", "w", "core")
        group = file.create_group("realization")
        
        realization.to_HDF(group)
        recalled_realization = self.model.recall_realization(group)
        
        file.close()
        
        self.assertEqual(expected_state, recalled_realization.expected_state)



class TestIndoorOffice(TestClusterDelayLine):
    
    def _init_model(self) -> IndoorOffice:
        return IndoorOffice(gain=self.gain, delay_normalization=self.delay_normalization, oxygen_absorption=self.oxygen_absorption)
    
    def _large_scale_states(self) -> list:
        return list(LOSState)
    
    def test_office_type(self) -> None:
        """Test different office types"""
        
        for office_type in list(OfficeType):
            self.model.office_type = office_type
            realization = self.model.realize()
            sample = realization.sample(self.alpha_device, self.beta_device, self.carrier_frequency, self.bandwidth)
            self._test_propagation(sample)
            
    def test_office_type_setget(self) -> None:
        """Office type getter should return setter argument"""
        
        self.model.office_type = OfficeType.OPEN
        self.assertEqual(OfficeType.OPEN, self.model.office_type)
            
    def test_hdf_serialization_expected_state(self) -> None:
        """Test HDF serialization with a fixed expected state"""
    
        expected_state = LOSState.LOS
        self.model.expected_state = expected_state
        
        realization = self.model.realize()
        
        file = File("test.hdf", "w", "core")
        group = file.create_group("realization")
        
        realization.to_HDF(group)
        recalled_realization = self.model.recall_realization(group)
        
        file.close()
        
        self.assertEqual(expected_state, recalled_realization.expected_state)


class TestRuralMacrocells(TestClusterDelayLine):

    def _init_model(self) -> RuralMacrocells:
        return RuralMacrocells(self.gain, self.delay_normalization, self.oxygen_absorption)
    
    def _large_scale_states(self) -> list:
        return list(O2IState)
    
    def test_hdf_serialization_expected_state(self) -> None:
        """Test HDF serialization with a fixed expected state"""
    
        expected_state = O2IState.LOS
        self.model.expected_state = expected_state
        
        realization = self.model.realize()
        
        file = File("test.hdf", "w", "core")
        group = file.create_group("realization")
        
        realization.to_HDF(group)
        recalled_realization = self.model.recall_realization(group)
        
        file.close()
        
        self.assertEqual(expected_state, recalled_realization.expected_state)


class TestUrbanMacrocells(TestClusterDelayLine):

    def _init_model(self) -> UrbanMacrocells:
        return UrbanMacrocells(self.gain, self.delay_normalization, self.oxygen_absorption)
    
    def _large_scale_states(self) -> list:
        return list(O2IState)
    
    def test_hdf_serialization_expected_state(self) -> None:
        """Test HDF serialization with a fixed expected state"""
    
        expected_state = LOSState.LOS
        self.model.expected_state = expected_state
        
        realization = self.model.realize()
        
        file = File("test.hdf", "w", "core")
        group = file.create_group("realization")
        
        realization.to_HDF(group)
        recalled_realization = self.model.recall_realization(group)
        
        file.close()
        
        self.assertEqual(expected_state, recalled_realization.expected_state)
    

class TestUrbanMicrocells(TestClusterDelayLine):

    def _init_model(self) -> UrbanMicrocells:
        return UrbanMicrocells(self.gain, self.delay_normalization, self.oxygen_absorption)
    
    def _large_scale_states(self) -> list:
        return list(O2IState)
    
    def test_hdf_serialization_expected_state(self) -> None:
        """Test HDF serialization with a fixed expected state"""
    
        expected_state = O2IState.LOS
        self.model.expected_state = expected_state
        
        realization = self.model.realize()
        
        file = File("test.hdf", "w", "core")
        group = file.create_group("realization")
        
        realization.to_HDF(group)
        recalled_realization = self.model.recall_realization(group)
        
        file.close()
        
        self.assertEqual(expected_state, recalled_realization.expected_state)
    
class TestCDL(TestCase):
    """Test static CDL models."""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.carrier_frequency = 1e9
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([0, 0, 0])))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([100, 100, 0])))
        self.bandwidth = 1e8
        self.gain = 0.98
        
        self.model = CDL(CDLType.E, 1e-8, 0.123, 29, gain=self.gain)
        
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(CDLType.E, self.model.model_type)
        self.assertEqual(1e-8, self.model.rms_delay)
        self.assertEqual(0.123, self.model.rayleigh_factor)
        self.assertEqual(29, self.model.decorrelation_distance)
        
    def test_rms_delay_setget(self) -> None:
        """RMS delay getter should return setter argument"""
        
        self.model.rms_delay = 1e-9
        self.assertEqual(1e-9, self.model.rms_delay)
        
    def test_rms_delay_validation(self) -> None:
        """RMS delay should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.model.rms_delay = -1e-9
            
    def test_rayleigh_factor_setget(self) -> None:
        """Rayleigh factor getter should return setter argument"""
        
        self.model.rayleigh_factor = 0.5
        self.assertEqual(0.5, self.model.rayleigh_factor)
        
    def test_rayleigh_factor_validation(self) -> None:
        """Rayleigh factor should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.model.rayleigh_factor = -0.5
            
    def test_decorrelation_distance_setget(self) -> None:
        """Decorrelation distance getter should return setter argument"""
        
        self.model.decorrelation_distance = 31
        self.assertEqual(31, self.model.decorrelation_distance)
        
    def test_decorrelation_distance_validation(self) -> None:
        """Decorrelation distance should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.model.decorrelation_distance = -30
            
    def test_realize(self) -> None:
        """Test a random realization"""
        
        for type in CDLType:
            model = CDL(type, 1e-8, 0.123, 29, gain=self.gain)
            
            realization = model.realize()
            sample = realization.sample(self.alpha_device, self.beta_device, self.carrier_frequency, self.bandwidth)
        
    def test_reciprocal_sample(self) -> None:
        """Test reciprocal sample"""
        
        realization = self.model.realize()
        sample = realization.sample(self.alpha_device, self.beta_device, self.carrier_frequency, self.bandwidth)
        
        reciprocal_sample = realization.reciprocal_sample(sample, self.alpha_device, self.beta_device)
        reciprocal_sample = realization.reciprocal_sample(sample, self.beta_device, self.alpha_device)
        
    def test_hdf_serialization(self) -> None:
        """Test the realization serialization to and from HDF"""
        
        expected_realization = self.model.realize()
        
        file = File("test.hdf", "w", "core")
        group = file.create_group("realization")
        
        expected_realization.to_HDF(group)
        recalled_realization = self.model.recall_realization(group)
        
        file.close()
        
        self.assertEqual(expected_realization.gain, recalled_realization.gain)


        
del TestClusterDelayLine
