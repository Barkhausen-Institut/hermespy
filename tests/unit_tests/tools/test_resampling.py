# -*- coding: utf-8 -*-
"""Test HermesPy resampling routines."""

from unittest import TestCase
from itertools import product

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.fft import fft, ifft, ifftshift, fftshift

from hermespy.tools import delay_resampling_matrix

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDelayResampling(TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_resampling_matrix_circular(self) -> None:
        """Make sure the resampling matrix has circular properties."""

        sampling_rate = 2e3
        num_sample_tests = [10, 100, 1000]
        delay_tests = np.array([0.0, 0.5, 1, 2, 0.3, 1.7]) / sampling_rate

        for num_samples, delay in product(num_sample_tests, delay_tests):
            positive_resampling_matrix = delay_resampling_matrix(sampling_rate, num_samples, delay)
            negative_resampling_matrix = delay_resampling_matrix(sampling_rate, positive_resampling_matrix.shape[0], -delay)

            circular_transformation = negative_resampling_matrix @ positive_resampling_matrix

            # Make sure the circular transformation is approximately unitary
            assert_almost_equal(np.ones(num_samples), np.diag(circular_transformation), decimal=1)

            # Make sure the power scales properly
            norm = np.linalg.norm(circular_transformation, axis=1)
            assert_almost_equal(np.ones(num_samples), norm, decimal=1)

    def test_resampling_matrix_dimensions(self) -> None:
        """The resampling matrix should scale its dimensions according to the arguments."""

        sampling_rate = 1e3
        delay = 5e4

        num_input_samples_tests = np.arange(4)
        num_output_samples_tests = np.arange(4)

        for num_input_samples, num_output_samples in product(num_input_samples_tests, num_output_samples_tests):
            resampling_matrix = delay_resampling_matrix(sampling_rate, num_input_samples, delay, num_output_samples)
            self.assertCountEqual([num_output_samples, num_input_samples], resampling_matrix.shape)
