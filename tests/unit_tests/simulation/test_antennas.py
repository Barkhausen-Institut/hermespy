# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.beamforming import (
    BeamformingReceiver,
    BeamformingTransmitter,
    CaponBeamformer,
    ConventionalBeamformer,
)
from hermespy.core import AntennaMode, Signal
from hermespy.simulation import (
    SimulatedAntenna,
    SimulatedAntennaArray,
    SimulatedCustomArray,
    SimulatedDevice,
    SimulatedDipole,
    SimulatedIdealAntenna,
    SimulatedLinearAntenna,
    SimulatedPatchAntenna,
    SimulatedUniformArray,
)

from unit_tests.utils import SimulationTestContext
from unit_tests.utils import assert_signals_equal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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

        signal = Signal.Create(
            np.random.normal(size=(2, 10)) + 1j * np.random.normal(size=(2, 10)), 1, 1
        )

        with self.assertRaises(ValueError):
            self.antenna.transmit(signal)

    def test_transmit(self) -> None:
        """Transmit routine should properly apply the antenna weight"""

        signal = Signal.Create(
            np.random.normal(size=(1, 10)) + 1j * np.random.normal(size=(1, 10)), 1, 1
        )
        weight = 2 + 3j
        expected_signal = weight * signal

        self.antenna.weight = weight
        transmitted_signal = self.antenna.transmit(signal)

        assert_signals_equal(self, expected_signal, transmitted_signal)

    def test_receive_validation(self) -> None:
        """Receive routine should raise ValueError if signal model has more than one stream"""

        signal = Signal.Create(
            np.random.normal(size=(2, 10)) + 1j * np.random.normal(size=(2, 10)), 1, 1
        )

        with self.assertRaises(ValueError):
            self.antenna.receive(signal)

    def test_receive(self) -> None:
        """Receive rotine should properly apply the antenna weight"""

        signal = Signal.Create(
            np.random.normal(size=(1, 10)) + 1j * np.random.normal(size=(1, 10)), 1, 1
        )
        weight = 2 + 3j
        expected_signal = weight * signal

        self.antenna.weight = weight
        received_signal = self.antenna.receive(signal)

        assert_signals_equal(self, expected_signal, received_signal)


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

    def test_transmit_validation(self) -> None:
        """Transmit routine should raise ValueError if the signal argument as an invalid number of streams"""

        signal = Signal.Create(
            np.random.normal(size=(10, 10)) + 1j * np.random.normal(size=(10, 10)), 1, 1
        )

        with self.assertRaises(ValueError):
            self.array.transmit(signal)

    def test_receive_validation(self) -> None:
        """Receive routine should raise a ValueError if the signal argument has an invalid number of streams"""

        signal = Signal.Create(
            np.random.normal(size=(10, 10)) + 1j * np.random.normal(size=(10, 10)), 1, 1
        )

        with self.assertRaises(ValueError):
            self.array.receive(signal)

    def test_visualize_far_field_pattern(self) -> None:
        """Far field pattern visualization should return a new figure"""

        signal = Signal.Create(
            np.random.normal(size=(self.array.num_transmit_antennas, 10))
            + 1j * np.random.normal(size=(self.array.num_transmit_antennas, 10)),
            carrier_frequency=1e9,
            sampling_rate=1e7,
        )

        with SimulationTestContext():
            _ = self.array.visualize_far_field_pattern(signal)

    def test_plot_pattern_validation(self) -> None:
        """Pattern plotting routine should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.array.plot_pattern(1e9, AntennaMode.TX, np.zeros((2, 2)))

        with self.assertRaises(ValueError):
            self.array.plot_pattern(1e9, "bad_argument", np.zeros((1, 2)))

    def test_caculate_power_validation(self) -> None:
        """Power Calculating Method should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.array.calculate_power(1e9, "bad_argument", np.zeros((1, 2)), aoi=np.zeros((2, 2)))

    def test_plot_pattern_weights(self) -> None:
        """Pattern plotting routine should generate a new figure"""

        with patch("matplotlib.pyplot.figure") as figure_mock:
            _ = self.array.plot_pattern(1e9, AntennaMode.TX, np.ones(self.array.num_transmit_antennas))
            figure_mock.assert_called_once()

            figure_mock.reset_mock()

            _ = self.array.plot_pattern(1e9, AntennaMode.RX, np.ones(self.array.num_receive_antennas))
            figure_mock.assert_called_once()

    def test_plot_pattern_transmit_beamformer(self) -> None:
        """Pattern plotting routine should generate a new figure given a transmit beamformer"""

        beamformer = ConventionalBeamformer()

        with patch("matplotlib.pyplot.figure") as figure_mock:
            _ = self.array.plot_pattern(1e9, beamformer)
            figure_mock.assert_called_once()

            figure_mock.reset_mock()

            _ = self.array.plot_pattern(1e9, beamformer, slice_list=[((0,0),(1, 0))])
            figure_mock.assert_called_once()

    def test_plot_pattern_receive_beamformer(self) -> None:
        """Pattern plotting routine should generate a new figure given a receive beamformer"""

        beamformer = ConventionalBeamformer()

        with patch("matplotlib.pyplot.figure") as figure_mock:
            _ = self.array.plot_pattern(1e9, beamformer)
            figure_mock.assert_called_once()

            figure_mock.reset_mock()

            _ = self.array.plot_pattern(1e9, beamformer, slice_list=[((0,0),(1, 0))])
            figure_mock.assert_called_once()


class TestSimulatedUniformArray(_TestSimulatedAntennas):
    """Test simulated uniform array model"""

    def setUp(self) -> None:
        self.array = SimulatedUniformArray(SimulatedIdealAntenna, 1, [2, 3, 4])


class TestSimulateCustomArray(_TestSimulatedAntennas):
    """Test simulated custom array model"""

    array: SimulatedCustomArray

    def setUp(self) -> None:
        self.antennas = [
            SimulatedIdealAntenna(AntennaMode.TX),
            SimulatedPatchAntenna(AntennaMode.RX),
            SimulatedLinearAntenna(AntennaMode.DUPLEX),
        ]
        self.array = SimulatedCustomArray(self.antennas)

    def test_add_antenna(self) -> None:
        """Test antenna addition"""

        expected_antenna = SimulatedIdealAntenna(AntennaMode.TX)
        self.array.add_antenna(expected_antenna)
        self.assertIn(expected_antenna, self.array.antennas)


del _TestSimulatedAntenna
del _TestSimulatedAntennas
