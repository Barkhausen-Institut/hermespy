# -*- coding: utf-8 -*-
"""Tests for the Quadriga Channel Matlab Interface to Hermes"""

from os import environ, path
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from h5py import File
from numpy.testing import assert_array_almost_equal

from hermespy.channel.quadriga.quadriga import QuadrigaChannel, QuadrigaChannelRealization, QuadrigaChannelSample
from hermespy.channel.quadriga.interface import QuadrigaInterface
from hermespy.channel.quadriga.matlab import QuadrigaMatlabInterface
from hermespy.channel.quadriga.octave  import QuadrigaOctaveInterface
from hermespy.channel.channel import LinkState
from hermespy.core import DenseSignal, Transformation
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import assert_signals_equal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestQuadrigaChannelSample(TestCase):
    """Test the quadriga channel sample"""
    
    def setUp(self) -> None:
        
        self.carrier_frequency = 1e9
        self.bandwidth = 1e6
        self.oversampling_factor = 2
        self.sampling_rate = self.bandwidth * self.oversampling_factor
        
        self.transmitter = SimulatedDevice(carrier_frequency=self.carrier_frequency, bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor)
        self.receiver = SimulatedDevice(carrier_frequency=self.carrier_frequency, bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor)
        
        self.rng = np.random.default_rng(42)
        self.path_gains = np.ones((1, 1, 5))
        self.path_delays = np.ones((1, 1, 5))
        self.gain = 1.0
        self.transmitter_state = self.transmitter.state(0)
        self.receiver_state = self.receiver.state(0)

        self.sample = QuadrigaChannelSample(self.path_gains, self.path_delays, self.gain, LinkState(self.transmitter_state, self.receiver_state, self.carrier_frequency, self.sampling_rate, 0.0))

    def test_propagate_state(self) -> None:
        """Propagation should result in a signal with the correct number of samples"""

        num_samples = 10
        signal = DenseSignal.FromNDArray(self.rng.normal(0, 1, size=(1, num_samples)) + 1j * self.rng.normal(0, 1, size=(1, num_samples)), self.sampling_rate, self.carrier_frequency)

        signal_propagation = self.sample.propagate(signal)
        state_propagation = self.sample.state(signal.num_samples, 1 + signal_propagation.num_samples - signal.num_samples).propagate(signal)

        assert_signals_equal(self, signal_propagation, state_propagation)


class TestQuadrigaChannelRealization(TestCase):
    """Test Quadriga channel realization"""

    def setUp(self) -> None:

        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), "..", "..", "..", "submodules", "quadriga", "quadriga_src"))
        self.interface = QuadrigaInterface(path_quadriga_src=path_quadriga_src)
        self.gain = 0.9876
        
        self.rng = np.random.default_rng(42)

        self.bandwidth = 1e6
        self.oversampling_factor = 2
        self.sampling_rate = self.bandwidth * self.oversampling_factor
        self.carrier_frequency = 1e9

        self.alpha_device = SimulatedDevice(bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor, carrier_frequency=self.carrier_frequency)
        self.beta_device = SimulatedDevice(bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor, carrier_frequency=self.carrier_frequency)

        self.delays = np.ones((1, 1, 5)) / self.sampling_rate
        self.cirs = self.rng.standard_normal((1, 1, 5, 10)) + 1j * self.rng.standard_normal((1, 1, 5, 10))

        self.realization = QuadrigaChannelRealization(self.interface, [], self.gain)

    def test_sample(self) -> None:
        """Test the Quadriga channel realization sampling"""
        
        with patch.object(self.interface, "sample_quadriga") as interface_mock:
            coefficients = self.rng.standard_normal((1, 1, 10)) + 1j * self.rng.standard_normal((1, 1, 10))
            delays = self.rng.normal(0, 1 / self.sampling_rate, size=(1, 1, 10))
            run_result = Mock()
            run_result.coefficients = coefficients
            run_result.delays = delays
            
            interface_mock.return_value = np.array([[run_result]])

            sample = self.realization.sample(self.alpha_device, self.beta_device)
            
        assert_array_almost_equal(coefficients, sample.path_gains)
        assert_array_almost_equal(delays, sample.path_delays)


class TestQuadrigaChannel(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        
        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), "..", "..", "..", "submodules", "quadriga", "quadriga_src"))
        self.interface = QuadrigaInterface(path_quadriga_src=path_quadriga_src)

        self.bandwidth = 1e6
        self.oversampling_factor = 2
        self.sampling_rate = self.bandwidth * self.oversampling_factor
        self.num_samples = 1000
        self.carrier_frequency = 1e9

        self.transmitter = SimulatedDevice(bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor, carrier_frequency=self.carrier_frequency)
        self.receiver = SimulatedDevice(bandwidth=self.bandwidth, oversampling_factor=self.oversampling_factor, carrier_frequency=self.carrier_frequency)
        self.transmitter.position = np.array([-500.0, 0.0, 0.0], dtype=float)
        self.receiver.position = np.array([500.0, 0.0, 0.0], dtype=float)

        self.channel = QuadrigaChannel(interface=self.interface)

    def test_realize(self) -> None:
        """Test the Quadriga channel realization"""

        realization = self.channel.realize()
        self.assertEqual(1.0, realization.gain)

    def test_serialization(self) -> None:
        """Test the Quadriga channel serialization"""

        environ["HERMES_QUADRIGA"] = path.abspath(path.join(path.dirname(__file__), "..", "..", "..", "submodules", "quadriga", "quadriga_src"))
        test_roundtrip_serialization(self, self.channel, {'random_mother'})


class TestQuadrigaInterface(TestCase):
    """Test the global quadriga interface"""

    def setUp(self) -> None:
        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), "..", "..", "..", "submodules", "quadriga", "quadriga_src"))
        self.interface = QuadrigaInterface(path_quadriga_src=path_quadriga_src)

    def test_environment_quadriga_path(self) -> None:
        """The interface should properly infer the quadriga source path from environment variables"""

        environ["HERMES_QUADRIGA"] = path.dirname(__file__)
        interface = QuadrigaInterface()

        self.assertEqual(path.dirname(__file__), interface.path_quadriga_src)

    def test_quadriga_path_validation(self) -> None:
        """The interface should properly validate the quadriga source path"""

        with self.assertRaises(ValueError):
            self.interface.path_quadriga_src = "testyyyyy"

    def test_antenna_kind_setget(self) -> None:
        """Antenna kind property getter should return setter argument"""

        self.interface.antenna_kind = "test"
        self.assertEqual("test", self.interface.antenna_kind)

    def test_scenario_label_setget(self) -> None:
        """Scenario label property getter should return setter argument"""

        self.interface.scenario_label = "test"
        self.assertEqual("test", self.interface.scenario_label)
        
    def test_sample_quadriga(self) -> None:
        """Test the interface sampling routine"""
        
        state = Mock()
        with patch.object(self.interface, "_run_quadriga") as run_mock:
            sample = self.interface.sample_quadriga(state)
            run_mock.assert_called_once()
        

    def test_run_quadriga(self) -> None:
        """Running quadriga should raise a NotImplementedError"""

        with self.assertRaises(NotImplementedError):
            self.interface._run_quadriga()


class TestQuadrigaMatlabInterface(TestCase):
    """Test the Quadriga Matlab Interface"""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)

        cls.start_matlab_patch = patch("hermespy.channel.quadriga.matlab.start_matlab")
        cls.matlab_patch = patch("hermespy.channel.quadriga.matlab.matlab")

        engine = Mock()
        quadriga_realization = Mock()
        quadriga_realization.path_impulse_responses = rng.standard_normal((1, 1, 10))
        quadriga_realization.tau = rng.normal(0, 1, size=(1, 1, 10))

        cirs = np.empty((1, 1), dtype=np.object_)
        cirs[0, 0] = quadriga_realization

        engine.workspace = {"cirs": cirs}
        cls.start_matlab_patch.return_value = engine

        cls.start_matlab_patch.start()
        cls.matlab_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.start_matlab_patch.stop()
        cls.matlab_patch.stop()

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), "..", "..", "..", "submodules", "quadriga", "quadriga_src"))
        self.interface = QuadrigaMatlabInterface(path_quadriga_src=path_quadriga_src)

    def test_run_quadriga(self) -> None:
        """Test the Matlab interface to Quadriga"""

        transmitter = SimulatedDevice(pose=Transformation.From_Translation(np.array([1, 2, 3])))
        receiver = SimulatedDevice(pose=Transformation.From_Translation(np.array([4, 5, 6])))
        channel = QuadrigaChannel(interface=self.interface)

        realization = channel.realize()
        self.assertIsInstance(realization, QuadrigaChannelRealization)


class TestQuadrigaOctaveInterface(TestCase):
    """Test the Quadriga Octave Interface"""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)

        cls.oct2py_patch = patch("hermespy.channel.quadriga.octave.Oct2Py")

        oct2py = Mock()
        cls.oct2py_patch.return_value = oct2py

        quadriga_realization = Mock()
        quadriga_realization.path_impulse_responses = rng.standard_normal((1, 1, 10))
        quadriga_realization.tau = rng.normal(0, 1, size=(1, 1, 10))

        cirs = np.empty((1, 1), dtype=np.object_)
        cirs[0, 0] = quadriga_realization

        oct2py.pull.return_value = cirs

        cls.oct2py_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.oct2py_patch.stop()

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), "..", "..", "..", "submodules", "quadriga", "quadriga_src"))
        self.interface = QuadrigaOctaveInterface(path_quadriga_src=path_quadriga_src)

    def test_run_quadriga(self) -> None:
        """Test the Oct2Py interface to Quadriga"""

        transmitter = SimulatedDevice(pose=Transformation.From_Translation(np.array([1, 2, 3])))
        receiver = SimulatedDevice(pose=Transformation.From_Translation(np.array([4, 5, 6])))
        channel = QuadrigaChannel(interface=self.interface)

        realization = channel.realize()
        self.assertIsInstance(realization, QuadrigaChannelRealization)
