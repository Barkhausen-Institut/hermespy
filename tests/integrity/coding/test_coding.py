import unittest
import os
import shutil

from scipy.io import loadmat
import numpy as np

import hermes
from tests.integrity.utils import compare_mat_files

class TestLdpcEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.settings_dir = os.path.join(
            "tests", "integrity", "coding", "settings", "ldpc")

        self.global_output_dir = os.path.join(
            "tests", "integrity", "coding", "results", "ldpc")

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)

    def test_matches_paper_verified_ber_curve_awgn(self) -> None:
        self.settings_dir = os.path.join(
            self.settings_dir, "awgn_channel_ldpc"
        )
        self.output_dir = os.path.join(
            self.global_output_dir, "awgn_channel_temp"
        )

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", self.output_dir
        ]

        hermes.hermes(self.arguments_hermes)
        expected_results_simulation = loadmat(
            os.path.join(self.global_output_dir, "awgn_channel", "statistics.mat"))
        results_simulation = loadmat(
            os.path.join(self.output_dir, "statistics.mat")
        )

        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation))

    def test_BER_lower_for_low_snrs_stochastic_channel(self) -> None:
        """It is expected that the BER will be higher for high SNR values with encoding
        than without encoding."""

        self.settings_dir = os.path.join(
            self.settings_dir, "stochastic_channel_ldpc"
        )
        self.output_dir = os.path.join(
            self.global_output_dir, "stochastic_channel")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", os.path.join(self.output_dir, "ldpc")]
        hermes.hermes(self.arguments_hermes)
        results_stochastic_channel_ldpc = loadmat(
            os.path.join(
                self.output_dir, "ldpc", "statistics.mat"))

        self.settings_dir = os.path.join(
            self.settings_dir, "..", "stochastic_channel")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", os.path.join(self.output_dir, "no_encoding")]
        hermes.hermes(self.arguments_hermes)
        results_stochastic_channel = loadmat(
            os.path.join(self.output_dir, "no_encoding", "statistics.mat"))

        self.assertTrue(np.all(
            results_stochastic_channel_ldpc["ber_mean"] > results_stochastic_channel["ber_mean"]
            )
        )
