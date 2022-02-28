from unittest import TestCase
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal


from hermespy.core import FloatingError
from hermespy.simulation.antenna import AntennaArray, IdealAntenna, UniformArray


class TestIdealAntenna(TestCase):
    """Test the ideal antenna model"""

    def setUp(self) -> None:

        self.antenna = IdealAntenna()

    def test_array_setget(self) -> None:
        """Array property getter should return setter argument."""

        array = Mock()
        self.antenna.array = array

        self.assertIs(array, self.antenna.array)

        self.antenna.array = array
        self.assertIs(array, self.antenna.array)

    def test_array_removal(self) -> None:
        """Removing an array should envoke the callback."""

        array = Mock()
        self.antenna.array = array
        self.antenna.array = None

        self.assertEqual(1, array.remove_antenna.call_count)


    def test_pos_set(self) -> None:
        """Setting an antenna element position should call the respective array callback"""

        array = Mock()
        self.antenna.array = array
        self.antenna.pos = [1, 2, 3]

        self.assertEqual(1, array.set_antenna_position.call_count)

    def test_pos_set_validation(self) -> None:
        """Setting the position of a floating antenna should raise an exception"""

        with self.assertRaises(FloatingError):
            self.antenna.pos = np.array([1, 2, 3])

    def test_transmit(self) -> None:
        """Transmitting should just be a stub returning the argument."""

        signal = Mock()
        self.assertIs(signal, self.antenna.transmit(signal))

    def test_receive(self) -> None:
        """Receiving should just be a stub returning the argument."""

        signal = Mock()
        self.assertIs(signal, self.antenna.receive(signal))

    def test_polarization(self) -> None:
        """Polarization should return unit polarization."""

        self.assertCountEqual((2 ** -.5, 2 ** -.5), self.antenna.polarization(0., 0.))

    def test_plot_polarization(self) -> None:
        """Calling the plot rountine should return a figure object."""

        self.assertIsInstance(self.antenna.plot_ploarization(), plt.Figure)

    def test_plot_gain(self) -> None:
        """Calling the plot rountine should return a figure object."""

        self.assertIsInstance(self.antenna.plot_gain(), plt.Figure)

class TestUniformArray(TestCase):
    """Test the Uniform array model."""

    def setUp(self) -> None:

        self.antenna = Mock()
        self.antenna.polarization.return_value = np.array([2 ** -.5, 2 ** -.5])
        self.spacing = 1e-3
        self.num_antennas = (10, 9, 8)

        self.array = UniformArray(self.antenna, self.spacing, self.num_antennas)


    def test_spacing_setget(self) -> None:
        """Spacing property getter should return setter argument."""

        spacing = 1.234
        self.array.spacing = spacing

        self.assertEqual(spacing, self.array.spacing)

    def test_spacing_validation(self) -> None:
        """Spacing property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.array.spacing = -1.

        with self.assertRaises(ValueError):
            self.array.spacing = 0.


    def test_num_antennas_setget(self) -> None:
        """The number of antennas property getter should return the proper antenna count."""

        self.array.num_antennas = 1
        self.assertCountEqual((1, 1, 1), self.array.num_antennas)


        self.array.num_antennas = (1, 2)
        self.assertCountEqual((1, 2, 1), self.array.num_antennas)

        self.array.num_antennas = (1, 2, 3)
        self.assertCountEqual((1, 2, 3), self.array.num_antennas)

    def test_num_antennas_validation(self) -> None:
        """The number of antennas property setter should raise a ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.array.num_antennas = (1, 2, 3, 4)

        with self.assertRaises(ValueError):
            self.array.num_antennas = (1, 2, -1)

    def test_topology(self) -> None:

        num_antennas = 5
        spacing = 1.
        expected_topology = np.zeros((num_antennas, 3), dtype=float)
        expected_topology[:, 0] = spacing * np.arange(num_antennas)

        self.array.num_antennas = num_antennas
        self.array.spacing = spacing

        assert_array_equal(expected_topology, self.array.topology)

    def test_polarization(self) -> None:
        """The polarization should compute the correct polarization array."""

        polariization = self.array.polarization(0., 0.)
        self.assertCountEqual((10 * 9 * 8, 2), polariization.shape)
        self.assertTrue(np.any(polariization == 2 ** -.5))

    def test_plot_topology(self) -> None:
        """Calling the plot rountine should return a figure object."""

        self.assertIsInstance(self.array.plot_topology(), plt.Figure)
