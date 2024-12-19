# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal

from hermespy.modem import Symbol, Symbols, StatedSymbols
from hermespy.modem.symbols import SymbolType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSymbol(TestCase):
    """Test class for single communication symbols"""

    def setUp(self) -> None:
        self.symbol = Symbol(1j)

    def test_properties(self) -> None:
        """Test the properties of the symbol"""

        self.assertEqual(self.symbol.value, 1j)
        self.assertEqual(SymbolType.DATA, self.symbol.flag)


class TestSymbols(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.raw_symbols = self.rng.normal(size=(3, 4, 5)) + 1j * self.rng.normal(size=(3, 4, 5))

        self.symbols = Symbols(self.raw_symbols)

    def test_init_validation(self) -> None:
        """Initialization should raise ValueError for invalid arguments"""

        with self.assertRaises(ValueError):
            Symbols(np.array([[[[1, 2, 3]]]]))

    def test_append_stream(self) -> None:
        """Appending a stream should yield the correct object"""

        initial_symbols = Symbols(self.raw_symbols[0, :, 0])

        new_symbols = self.rng.normal(size=(4)) + 1j * self.rng.normal(size=(4))
        initial_symbols.append_stream(new_symbols)
        self.assertEqual(2, initial_symbols.num_streams)

        new_symbols = self.rng.normal(size=(1, 4)) + 1j * self.rng.normal(size=(1, 4))
        initial_symbols.append_stream(new_symbols)
        self.assertEqual(3, initial_symbols.num_streams)

    def test_append_stream_validation(self) -> None:
        """Appending a stream should raise ValueError for invalid arguments"""

        with self.assertRaises(ValueError):
            self.symbols.append_stream(np.array([[[[1, 2, 3]]]]))

        with self.assertRaises(ValueError):
            self.symbols.append_stream(np.zeros((3, 4, 6)))

        with self.assertRaises(ValueError):
            self.symbols.append_stream(np.zeros((3, 5, 5)))

    def test_append_symbols_vector(self) -> None:
        """Append a vector of symbols should yield the correct object"""

        initial_symbols = Symbols(self.raw_symbols.flatten())

        appended_symbols = self.raw_symbols[0, :, 0]
        initial_symbols.append_symbols(appended_symbols)

        assert_array_equal(appended_symbols, initial_symbols.raw[0, -appended_symbols.size :, 0])

    def test_append_symbols(self) -> None:
        """Appending symbols should yield the correct object"""

        initial_symbols = Symbols(self.raw_symbols[0, :, 0])

        new_symbols = Symbols(self.raw_symbols[0, :, 0])
        initial_symbols.append_symbols(new_symbols)
        self.assertEqual(8, initial_symbols.num_blocks)

        initial_symbols.append_symbols(self.raw_symbols[[0], :, 0])
        self.assertEqual(12, initial_symbols.num_blocks)

    def test_append_symbols_validation(self) -> None:
        """Appending symbols should raise ValueError for invalid arguments"""

        with self.assertRaises(ValueError):
            self.symbols.append_symbols(np.array([[[[1, 2, 3]]]]))

        with self.assertRaises(ValueError):
            self.symbols.append_symbols(Symbols(np.zeros((4, 4, 6))))

    def test_raw_validation(self) -> None:
        """Setting raw symbols should raise ValueError for invalid arguments"""

        with self.assertRaises(ValueError):
            self.symbols.raw = np.zeros(2)

    def test_raw_setget(self) -> None:
        """Setting raw symbols should yield the correct object"""

        raw_symbols = self.rng.normal(size=(3, 4, 5)) + 1j * self.rng.normal(size=(3, 4, 5))
        self.symbols.raw = raw_symbols

        np.testing.assert_array_equal(raw_symbols, self.symbols.raw)

    def test_copy(self) -> None:
        """Copying should yield the correct object"""

        copy = self.symbols.copy()
        np.testing.assert_array_equal(self.symbols.raw, copy.raw)

    def test_item_setget(self) -> None:
        """Item getter should return setter argument"""

        expected_symbol = 1j + 2
        self.symbols[1, 2, 3] = expected_symbol
        self.assertEqual(expected_symbol, self.symbols[1, 2, 3].raw)

        self.symbols[0, 1, 2] = Symbols(expected_symbol)
        self.assertEqual(expected_symbol, self.symbols[0, 1, 2].raw)

    def test_plot_constellation(self) -> None:
        """Plotting the constellation should yield the correct plot"""

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            fig_mock = MagicMock()
            ax_mock = MagicMock()
            subplots_mock.return_value = (fig_mock, ax_mock)

            self.symbols.plot_constellation()

            subplots_mock.assert_called_once()
            fig_mock.suptitle.assert_called_once()
            subplots_mock.reset_mock()
            fig_mock.reset_mock()
            ax_mock.reset_mock()

            axes = np.empty((1, 1), dtype=np.object_)
            axes[0, 0] = ax_mock
            self.symbols.plot_constellation(axes=axes)
            ax_mock.scatter.assert_called()

    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""

        symbols: Symbols = None

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "a") as file:
                group = file.create_group("testgroup")
                self.symbols.to_HDF(group)

            with File(file_location, "r") as file:
                group = file["testgroup"]
                symbols = self.symbols.from_HDF(group)

        np.testing.assert_array_equal(self.raw_symbols, symbols.raw)


class TestStatedSymbols(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.raw_symbols = self.rng.normal(size=(3, 4, 5)) + 1j * self.rng.normal(size=(3, 4, 5))
        self.raw_states = self.rng.normal(size=(3, 2, 4, 5)) + 1j * self.rng.normal(size=(3, 2, 4, 5))

        self.symbols = StatedSymbols(self.raw_symbols, self.raw_states)

    def test_states_validation(self) -> None:
        """Setting states should raise ValueError for invalid arguments"""

        with self.assertRaises(ValueError):
            self.symbols.states = np.zeros(2)

        with self.assertRaises(ValueError):
            self.symbols.states = np.zeros((4, 2, 4, 5))

        with self.assertRaises(ValueError):
            self.symbols.states = np.zeros((3, 2, 5, 5))

        with self.assertRaises(ValueError):
            self.symbols.states = np.zeros((3, 2, 4, 6))

    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""

        symbols: StatedSymbols = None

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "a") as file:
                group = file.create_group("testgroup")
                self.symbols.to_HDF(group)

            with File(file_location, "r") as file:
                group = file["testgroup"]
                symbols = self.symbols.from_HDF(group)

        np.testing.assert_array_equal(self.raw_symbols, symbols.raw)
        np.testing.assert_array_equal(self.raw_states, symbols.states)
