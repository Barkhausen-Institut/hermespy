# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.beamforming import BeamformingReceiver, BeamformingTransmitter, CaponBeamformer, ConventionalBeamformer
from hermespy.core import AntennaMode, Signal
from hermespy.simulation import RfChain, SimulatedAntenna, SimulatedAntennaPort, SimulatedAntennaArray, SimulatedCustomArray, SimulatedDevice, SimulatedDipole, SimulatedIdealAntenna, SimulatedLinearAntenna, SimulatedPatchAntenna, SimulatedUniformArray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulatedAntennaPort(TestCase):
    """Test simulated antenna port class"""

    def setUp(self) -> None:

        self.antennas = [
            SimulatedIdealAntenna(AntennaMode.TX),
            SimulatedIdealAntenna(AntennaMode.RX),
            SimulatedIdealAntenna(AntennaMode.DUPLEX),
        ]
        self.rf_chain = RfChain()
        self.port = SimulatedAntennaPort(self.antennas, rf_chain=self.rf_chain)

    def test_init(self) -> None:
        """Initialization arguments should be stored correctly"""

        self.assertSequenceEqual(self.port.antennas, self.antennas)
        self.assertIs(self.port.rf_chain, self.rf_chain)

    def test_rf_chain_setget(self) -> None:
        """RF chain porperty getter should return setter argument"""

        array = Mock(spec=SimulatedAntennaArray)
        self.port.array = array

        rf_chain = Mock(spec=RfChain)
        self.port.rf_chain = rf_chain

        self.assertIs(self.port.rf_chain, rf_chain)
        array.rf_chain_modified.assert_called_once()


class _TestSimulatedAntenna(TestCase):
    """Test simulated antenna model"""

    antenna: SimulatedAntenna

    def test_weight_setget(self) -> None:
        """Weight property getter should return setter argument"""

        weight = Mock()
        self.antenna.weight = weight

        self.assertIs(self.antenna.weight, weight)

    def test_transmit_validation(self) -> None:
        """Transmit routine should raise ValueError if signal model has more than one stream"""

        signal = Signal.Create(np.random.normal(size=(2, 10)) + 1j * np.random.normal(size=(2, 10)), 1, 1)

        with self.assertRaises(ValueError):
            self.antenna.transmit(signal)

    def test_transmit(self) -> None:
        """Transmit routine should properly apply the antenna weight"""

        signal = Signal.Create(np.random.normal(size=(1, 10)) + 1j * np.random.normal(size=(1, 10)), 1, 1)
        weight = 2+3j
        expected_samples = weight * signal.getitem()

        self.antenna.weight = weight
        transmitted_signal = self.antenna.transmit(signal)

        assert_array_almost_equal(expected_samples, transmitted_signal.getitem())

    def test_receive_validation(self) -> None:
        """Receive routine should raise ValueError if signal model has more than one stream"""

        signal = Signal.Create(np.random.normal(size=(2, 10)) + 1j * np.random.normal(size=(2, 10)), 1, 1)

        with self.assertRaises(ValueError):
            self.antenna.receive(signal)

    def test_receive(self) -> None:
        """Receive rotine should properly apply the antenna weight"""

        signal = Signal.Create(np.random.normal(size=(1, 10)) + 1j * np.random.normal(size=(1, 10)), 1, 1)
        weight = 2+3j
        expected_samples = weight * signal.getitem()

        self.antenna.weight = weight
        received_signal = self.antenna.receive(signal)

        assert_array_almost_equal(expected_samples, received_signal.getitem())


class TestSimulatedDipole(_TestSimulatedAntenna):
    """Test simulated dipole antenna model"""

    def setUp(self) -> None:
        self.antenna = SimulatedDipole(AntennaMode.DUPLEX)


class TestSimulatedIdealAntenna(_TestSimulatedAntenna):
    """Test simulated ideal antenna model"""

    def setUp(self) -> None:
        self.antenna = SimulatedIdealAntenna(AntennaMode.DUPLEX)


class TestSimulatedLinearAntenna(_TestSimulatedAntenna):
    """Test simulated linear antenna model"""

    def setUp(self) -> None:
        self.antenna = SimulatedLinearAntenna(AntennaMode.DUPLEX)


class TestSimulatedPatchAntenna(_TestSimulatedAntenna):
    """Test simulated patch antenna model"""

    def setUp(self) -> None:
        self.antenna = SimulatedPatchAntenna(AntennaMode.DUPLEX)


class _TestSimulatedAntennas(TestCase):

    array: SimulatedAntennaArray

    def test_rf_transmit_chains_caching(self) -> None:
        """RF transmit chains should be properly cached"""

        default_chain = Mock(spec=RfChain)
        transmit_chains = self.array._rf_transmit_chains(default_chain)
        cached_transmit_chains = self.array._rf_transmit_chains(default_chain)

        self.assertDictEqual(transmit_chains, cached_transmit_chains)

    def test_rf_receive_chains_caching(self) -> None:
        """RF receive chains should be properly cached"""

        default_chain = Mock(spec=RfChain)
        receive_chains = self.array._rf_receive_chains(default_chain)
        cached_receive_chains = self.array._rf_receive_chains(default_chain)

        self.assertDictEqual(receive_chains, cached_receive_chains)

    def test_transmit_validation(self) -> None:
        """Transmit routine should raise ValueError if the signal argument as an invalid number of streams"""

        signal = Signal.Create(np.random.normal(size=(10, 10)) + 1j * np.random.normal(size=(10, 10)), 1, 1)

        with self.assertRaises(ValueError):
            self.array.transmit(signal, Mock())

    def test_receive_validation(self) -> None:
        """Receive routine should raise a ValueError if the signal argument has an invalid number of streams"""

        signal = Signal.Create(np.random.normal(size=(10, 10)) + 1j * np.random.normal(size=(10, 10)), 1, 1)

        with self.assertRaises(ValueError):
            self.array.receive(signal, Mock())

    def test_receive_perfect(self) -> None:
        """Test receive routine without imperfections"""

        rng = np.random.default_rng(0)
        signal = Signal.Create(rng.normal(size=(self.array.num_receive_ports, 10)) + 1j * rng.normal(size=(self.array.num_receive_ports, 10)), 1, 1)
        rf_chain = RfChain()

        expected_samples = signal.getitem()
        received_signal = self.array.receive(signal, rf_chain)

        assert_array_almost_equal(expected_samples, received_signal.getitem())

    def test_receive_weights(self) -> None:
        """Receive routine should properly apply the antenna weights"""

        rng = np.random.default_rng(0)
        signal = Signal.Create(rng.normal(size=(self.array.num_receive_antennas, 10)) + 1j * rng.normal(size=(self.array.num_receive_antennas, 10)), 1, 1)
        rf_chain = RfChain()

        weights = rng.normal(size=self.array.num_receive_antennas) + 1j * rng.normal(size=self.array.num_receive_antennas)
        for antenna, weight in zip(self.array.receive_antennas, weights):
            antenna.weight = weight

        expected_samples = weights[:, None] * signal.getitem()
        received_signal = self.array.receive(signal, rf_chain)

        assert_array_almost_equal(expected_samples, received_signal.getitem())

    def test_receive_coupling(self) -> None:
        """Mutual coupling should be properly applied to the received signal"""

        rng = np.random.default_rng(0)
        signal = Signal.Create(rng.normal(size=(self.array.num_receive_ports, 10)) + 1j * rng.normal(size=(self.array.num_receive_ports, 10)), 1, 1)
        rf_chain = RfChain()

        coupling = Mock()
        coupling.receive.side_effect = lambda x: x

        _ = self.array.receive(signal, rf_chain, coupling_model=coupling)
        coupling.receive.assert_called_once()

    def test_receive_leakage_validation(self) -> None:
        """Receive should raise a ValueError if leakge argument has invalid number of streams"""

        rng = np.random.default_rng(0)
        signal = Signal.Create(rng.normal(size=(self.array.num_receive_ports, 10)) + 1j * rng.normal(size=(self.array.num_receive_ports, 10)), 1, 1)
        rf_chain = RfChain()

        leakage = Mock(spec=Signal)
        leakage.num_streams = 123

        with self.assertRaises(ValueError):
            self.array.receive(signal, rf_chain, leakage)

    def test_receive_leakage(self) -> None:
        """Leakge should be properly applied to the received signal"""

        rng = np.random.default_rng(0)
        signal = Signal.Create(rng.normal(size=(self.array.num_receive_ports, 10)) + 1j * rng.normal(size=(self.array.num_receive_ports, 10)), 1, 1)
        rf_chain = RfChain()

        leakage = Signal.Create(rng.normal(size=(self.array.num_receive_ports, 10)) + 1j * rng.normal(size=(self.array.num_receive_ports, 10)), 1, 1)

        expected_samples = signal.getitem() + leakage.getitem()
        received_signal = self.array.receive(signal, rf_chain, leakage)

        assert_array_almost_equal(expected_samples, received_signal.getitem())

    def test_receive_rf_chains(self) -> None:
        """RF chains should only be called once per receive call"""

        rf_mocks = [Mock(spec=RfChain) for _ in self.array.transmit_ports]
        for port, mock in zip(self.array.receive_ports, rf_mocks):
            port.rf_chain = mock
            mock.receive.side_effect = lambda x: x

        rng = np.random.default_rng(0)
        signal = Signal.Create(rng.normal(size=(self.array.num_receive_ports, 10)) + 1j * rng.normal(size=(self.array.num_receive_ports, 10)), 1, 1)
        default_rf_chain = Mock(spec=RfChain)

        _ = self.array.receive(signal, default_rf_chain)

        for mock in rf_mocks:
            mock.receive.assert_called_once()

        default_rf_chain.receive.assert_not_called()

    def test_analog_digital_conversion(self) -> None:
        """Test analog to digital conversion"""

        default_rf_chain = RfChain()
        rng = np.random.default_rng(0)
        test_signal = Signal.Create(rng.normal(size=(self.array.num_receive_ports, 10)) + 1j * rng.normal(size=(self.array.num_receive_ports, 10)), 1., 1.)

        quantized_signal = self.array.analog_digital_conversion(test_signal, default_rf_chain, 10)
        self.assertEqual(quantized_signal.num_samples, test_signal.num_samples)
        self.assertEqual(quantized_signal.num_streams, test_signal.num_streams)

    def test_plot_pattern_validation(self) -> None:
        """Pattern plotting routine should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.array.plot_pattern(1e9, AntennaMode.TX, np.zeros((2, 2)))

        with self.assertRaises(ValueError):
            self.array.plot_pattern(1e9, 'bad_argument', np.zeros((1, 2)))

    def test_plot_pattern_weights(self) -> None:
        """Pattern plotting routine should generate a new figure"""

        with patch('matplotlib.pyplot.figure') as figure_mock:
            _ = self.array.plot_pattern(1e9, AntennaMode.TX, np.ones(self.array.num_transmit_ports))
            figure_mock.assert_called_once()

            figure_mock.reset_mock()

            _ = self.array.plot_pattern(1e9, AntennaMode.RX, np.ones(self.array.num_receive_ports))
            figure_mock.assert_called_once()

    def test_plot_pattern_transmit_beamformer(self) -> None:
        """Pattern plotting routine should generate a new figure given a transmit beamformer"""

        signal = Signal.Create(np.random.normal(size=(1, 10)) + 1j * np.random.normal(size=(1, 10)), 1, 1)
        beamformer = ConventionalBeamformer()
        operator = BeamformingTransmitter(signal, beamformer)
        operator.device = SimulatedDevice(antennas=self.array)

        with patch('matplotlib.pyplot.figure') as figure_mock:
            _ = self.array.plot_pattern(1e9, beamformer)
            figure_mock.assert_called_once()

    def test_plot_pattern_receive_beamformer(self) -> None:
        """Pattern plotting routine should generate a new figure given a receive beamformer"""

        beamformer = CaponBeamformer(1e-3)
        operator = BeamformingReceiver(beamformer, 1, 1.)
        operator.device = SimulatedDevice(antennas=self.array)

        with patch('matplotlib.pyplot.figure') as figure_mock:
            _ = self.array.plot_pattern(1e9, beamformer)
            figure_mock.assert_called_once()


class TestSimulatedUniformArray(_TestSimulatedAntennas):
    """Test simulated uniform array model"""

    def setUp(self) -> None:
        self.array = SimulatedUniformArray(SimulatedIdealAntenna, 1, [2, 3, 4])


class TestSimulateCustomArray(_TestSimulatedAntennas):
    """Test simulated custom array model"""

    def setUp(self) -> None:
        self.antennas = [
            SimulatedIdealAntenna(AntennaMode.TX),
            SimulatedPatchAntenna(AntennaMode.RX),
            SimulatedLinearAntenna(AntennaMode.DUPLEX),
        ]
        self.array = SimulatedCustomArray(self.antennas)

    def test_add_port(self) -> None:
        """Test port addition"""

        with patch.object(self.array, 'rf_chain_modified') as chain_mock:

            expected_port = SimulatedAntennaPort()
            self.array.add_port(expected_port)

            chain_mock.assert_called_once()
            self.assertIn(expected_port, self.array.ports)

    def test_remove_port(self) -> None:
        """Test port removal"""

        with patch.object(self.array, 'rf_chain_modified') as chain_mock:

            expected_port = SimulatedAntennaPort()

            self.array.add_port(expected_port)
            chain_mock.reset_mock()

            self.array.remove_port(expected_port)

            self.assertNotIn(expected_port, self.array.ports)

    def test_add_antenna(self) -> None:
        """Test antenna addition"""

        with patch.object(self.array, 'rf_chain_modified') as chain_mock:

            expected_antenna = SimulatedIdealAntenna(AntennaMode.TX)
            self.array.add_antenna(expected_antenna)
            chain_mock.assert_called()
            self.assertIn(expected_antenna, self.array.antennas)


del _TestSimulatedAntenna
del _TestSimulatedAntennas
