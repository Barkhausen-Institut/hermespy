# -*- coding: utf-8 -*-
"""Tests for the Quadriga Channel Matlab Interface to Hermes"""

from os import environ, path
from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_almost_equal, assert_array_equal

from hermespy.channel import QuadrigaChannel, QuadrigaChannelRealization
from hermespy.channel.quadriga_interface import QuadrigaInterface
from hermespy.channel.quadriga_interface_matlab import QuadrigaMatlabInterface
from hermespy.channel.quadriga_interface_octave import QuadrigaOctaveInterface
from hermespy.core import Signal, Transformation
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestQuadrigaChannelRealization(TestCase):
    """Test Quadriga channel realization"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)

        self.sampling_rate = 1e6
        self.carrier_frequency = 1e9
        
        self.alpha_device = SimulatedDevice(sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency)
        self.beta_device = SimulatedDevice(sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency)

        self.delays = np.ones((1, 1, 5)) / self.sampling_rate
        self.cirs = self.rng.standard_normal((1, 1, 5, 10)) + 1j * self.rng.standard_normal((1, 1, 5, 10))

        self.realization = QuadrigaChannelRealization(self.alpha_device, self.beta_device, 1., self.cirs, self.delays)

    def test_propagate_validation(self) -> None:
        """Propagation should raise a ValueError for invalid signal arguments"""

        num_samples = 15
        signal = Signal(self.rng.normal(0, 1, size=(1, num_samples)) + 1j * self.rng.normal(0, 1, size=(1, num_samples)), self.sampling_rate)

        with self.assertRaises(ValueError):
            _ = self.realization.propagate(signal)

    def test_propagate_state(self) -> None:
        """Propagation should result in a signal with the correct number of samples"""
        
        num_samples = 10
        signal = Signal(self.rng.normal(0, 1, size=(1, num_samples)) + 1j * self.rng.normal(0, 1, size=(1, num_samples)), self.sampling_rate)
        
        signal_propagation = self.realization.propagate(signal)
        state_propagation = self.realization.state(self.alpha_device, self.beta_device, 0., self.sampling_rate, signal.num_samples, 1 + signal_propagation.signal.num_samples - signal.num_samples).propagate(signal)

        assert_array_almost_equal(signal_propagation.signal.samples, state_propagation.samples)

    def test_hdf_serialization(self) -> None:
        """Test serialization from and to HDF"""
        
        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('g')
        
        self.realization.to_HDF(group)
        recalled_realization = QuadrigaChannelRealization.From_HDF(group, self.alpha_device, self.beta_device)
        
        file.close()
        
        assert_array_equal(self.realization.path_gains, recalled_realization.path_gains)
        assert_array_equal(self.realization.path_delays, recalled_realization.path_delays)
        self.assertIs(self.realization.alpha_device, recalled_realization.alpha_device)
        self.assertIs(self.realization.beta_device, recalled_realization.beta_device)
        self.assertEqual(self.realization.gain, recalled_realization.gain)


class TestQuadrigaChannel(TestCase):

    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), '..', '..', '..', 'submodules', 'quadriga', 'quadriga_src'))
        self.interface = QuadrigaInterface(path_quadriga_src=path_quadriga_src)
                                           
        self.sampling_rate = 1e6
        self.num_samples = 1000
        self.carrier_frequency = 1e9

        self.transmitter = SimulatedDevice(sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency)
        self.receiver = SimulatedDevice(sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency)
        self.transmitter.position = np.array([-500., 0., 0.], dtype=float)
        self.receiver.position = np.array([500., 0., 0.], dtype=float)

        self.channel = QuadrigaChannel(self.transmitter, self.receiver, interface=self.interface)

        """Quadriga channel should be properly registered at the interface"""

        self.assertTrue(self.interface.channel_registered(self.channel))

    def test_realize(self) -> None:
        """Test the Quadriga channel realization"""
        
        with patch('hermespy.channel.quadriga_interface.QuadrigaInterface._run_quadriga') as run_mock:
            
            quadriga_realization = Mock()
            quadriga_realization.path_impulse_responses = self.rng.standard_normal((1, 1, self.num_samples))
            quadriga_realization.tau = self.rng.normal(0, 1/self.sampling_rate, size=(1, 1, self.num_samples))
            
            cirs = np.empty((1, 1), dtype=np.object_)
            cirs[0, 0] = quadriga_realization
            run_mock.return_value = cirs

            realization = self.channel.realize()

        self.assertIs(self.transmitter, realization.alpha_device)
        self.assertIs(self.receiver, realization.beta_device)
        self.assertEqual(1., realization.gain)
        
    def test_recall_realization(self) -> None:
        """Test the Quadriga channel realization recall"""
        
        with patch.object(self.interface, 'get_impulse_response') as interface_mock:
            
            path_impulse_responses = self.rng.standard_normal((1, 1, self.num_samples)) + 1j* self.rng.standard_normal((1, 1, self.num_samples))
            tau = self.rng.normal(0, 1/self.sampling_rate, size=(1, 1, self.num_samples))
            interface_mock.return_value = path_impulse_responses, tau

            file = File('test.h5', 'w', driver='core', backing_store=False)
            group = file.create_group('g')
            
            expected_realization = self.channel.realize()
            expected_realization.to_HDF(group)
            recalled_realization = self.channel.recall_realization(group)
            
            file.close()

            self.assertIsInstance(recalled_realization, type(expected_realization))

    def test_yaml_serialization(self) -> None:
        """Test the Quadriga Channel YAML serialization"""
        
        test_yaml_roundtrip_serialization(self, self.channel)


class TestQuadrigaInterface(TestCase):
    """Test the global quadriga interface"""

    def setUp(self) -> None:

        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), '..', '..', '..', 'submodules', 'quadriga', 'quadriga_src'))
        self.interface = QuadrigaInterface(path_quadriga_src=path_quadriga_src)

    def test_environment_quadriga_path(self) -> None:
        """The interface should properly infer the quadriga source path from environment variables"""

        environ['HERMES_QUADRIGA'] = path.dirname(__file__)
        interface = QuadrigaInterface()

        self.assertEqual(path.dirname(__file__), interface.path_quadriga_src)

    def test_set_global_instance(self) -> None:
        """The global instance should be settable"""

        interface = QuadrigaInterface()
        QuadrigaInterface.SetGlobalInstance(interface)

        self.assertEqual(interface, QuadrigaInterface.GlobalInstance())
        self.assertTrue(QuadrigaInterface.GlobalInstanceExists())
        
    def test_quadriga_path_validation(self) -> None:
        """The interface should properly validate the quadriga source path"""
        
        with self.assertRaises(ValueError):
            self.interface.path_quadriga_src = 'testyyyyy'

    def test_antenna_kind_setget(self) -> None:
        """Antenna kind property getter should return setter argument"""
        
        self.interface.antenna_kind = 'test'
        self.assertEqual('test', self.interface.antenna_kind)
        
    def test_scenario_label_setget(self) -> None:
        """Scenario label property getter should return setter argument"""
        
        self.interface.scenario_label = 'test'
        self.assertEqual('test', self.interface.scenario_label)
        
    def test_register_channel(self) -> None:
        """Channel should be properly registered"""
        
        channel = QuadrigaChannel(interface=self.interface)
        
        self.assertTrue(self.interface.channel_registered(channel))
        self.assertCountEqual([channel], self.interface.channels)
        
        with self.assertRaises(ValueError):
            self.interface.register_channel(channel)

        self.interface.unregister_channel(channel)
        self.assertFalse(self.interface.channel_registered(channel))

    def test_get_impulse_response_validation(self) -> None:
        """The interface should properly validate the impulse response arguments"""
        
        with self.assertRaises(ValueError):
            self.interface.get_impulse_response(Mock())
            
        transmitter = SimulatedDevice()
        receiver = SimulatedDevice()
        channel = QuadrigaChannel(interface=self.interface, alpha_device=transmitter, beta_device=receiver)
        
        with self.assertRaises(RuntimeError):
            self.interface.get_impulse_response(channel)
            
    def test_get_impulse_response(self) -> None:
        """Test impulse response generation"""
        
        transmitter = SimulatedDevice(pose=Transformation.From_Translation(np.array([1, 2, 3])))
        receiver = SimulatedDevice(pose=Transformation.From_Translation(np.array([4, 5, 6])))
        channel = QuadrigaChannel(interface=self.interface, alpha_device=transmitter, beta_device=receiver)
        
        with patch('hermespy.channel.quadriga_interface.QuadrigaInterface._run_quadriga') as run_mock:
            
            quadriga_realization = Mock()
            quadriga_realization.path_impulse_responses = Mock()
            quadriga_realization.tau = Mock()
            
            cirs = np.empty((1, 1), dtype=np.object_)
            cirs[0, 0] = quadriga_realization
            
            run_mock.return_value = cirs
            
            impulse_responses, tau = self.interface.get_impulse_response(channel)
            self.assertIs(impulse_responses, quadriga_realization.path_impulse_responses)
            self.assertIs(tau, quadriga_realization.tau)
            
            # Test relaunch
            impulse_responses, tau = self.interface.get_impulse_response(channel)
            self.assertIs(impulse_responses, quadriga_realization.path_impulse_responses)
            self.assertIs(tau, quadriga_realization.tau)

    def test_run_quadriga(self) -> None:
        """Running quadriga should raise a NotImplementedError"""
        
        with self.assertRaises(NotImplementedError):
            self.interface._run_quadriga()
            
            
class TestQuadrigaMatlabInterface(TestCase):
    """Test the Quadriga Matlab Interface"""
    
    @classmethod
    def setUpClass(cls) -> None:
        
        rng = np.random.default_rng(42)
        
        cls.start_matlab_patch = patch('hermespy.channel.quadriga_interface_matlab.start_matlab')
        cls.matlab_patch = patch('hermespy.channel.quadriga_interface_matlab.matlab')

        engine = Mock()
        quadriga_realization = Mock()
        quadriga_realization.path_impulse_responses =  rng.standard_normal((1, 1, 10))
        quadriga_realization.tau = rng.normal(0, 1, size=(1, 1, 10))
        
        cirs = np.empty((1, 1), dtype=np.object_)
        cirs[0, 0] = quadriga_realization
        
        engine.workspace = {'cirs': cirs}
        cls.start_matlab_patch.return_value = engine

        cls.start_matlab_patch.start()
        cls.matlab_patch.start()
        
    @classmethod
    def tearDownClass(cls) -> None:
        
        cls.start_matlab_patch.stop()
        cls.matlab_patch.stop()

    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), '..', '..', '..', 'submodules', 'quadriga', 'quadriga_src'))
        self.interface = QuadrigaMatlabInterface(path_quadriga_src=path_quadriga_src)

    def test_run_quadriga(self) -> None:
        """Test the Matlab interface to Quadriga"""
        
        transmitter = SimulatedDevice(pose=Transformation.From_Translation(np.array([1, 2, 3])))
        receiver = SimulatedDevice(pose=Transformation.From_Translation(np.array([4, 5, 6])))
        channel = QuadrigaChannel(interface=self.interface, alpha_device=transmitter, beta_device=receiver)
        
        realization = channel.realize()
        self.assertIsInstance(realization, QuadrigaChannelRealization)


class TestQuadrigaOctaveInterface(TestCase):
    """Test the Quadriga Octave Interface"""
    
    @classmethod
    def setUpClass(cls) -> None:
        
        rng = np.random.default_rng(42)
        

        cls.oct2py_patch = patch('hermespy.channel.quadriga_interface_octave.Oct2Py')

        oct2py = Mock()
        cls.oct2py_patch.return_value = oct2py

        quadriga_realization = Mock()
        quadriga_realization.path_impulse_responses =  rng.standard_normal((1, 1, 10))
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
        
        path_quadriga_src = path.abspath(path.join(path.dirname(__file__), '..', '..', '..', 'submodules', 'quadriga', 'quadriga_src'))
        self.interface = QuadrigaOctaveInterface(path_quadriga_src=path_quadriga_src)

    def test_run_quadriga(self) -> None:
        """Test the Oct2Py interface to Quadriga"""
        
        transmitter = SimulatedDevice(pose=Transformation.From_Translation(np.array([1, 2, 3])))
        receiver = SimulatedDevice(pose=Transformation.From_Translation(np.array([4, 5, 6])))
        channel = QuadrigaChannel(interface=self.interface, alpha_device=transmitter, beta_device=receiver)
        
        realization = channel.realize()
        self.assertIsInstance(realization, QuadrigaChannelRealization)
