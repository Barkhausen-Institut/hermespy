import unittest
import os
import shutil

from scipy.io import loadmat

import hermes
from tests.integrity.utils import compare_mat_files, compare_theory_simulation_errors


class TestBpskQam(unittest.TestCase):
    def setUp(self) -> None:
        # the path is relative to os.getcwd(), i.e. sys.path()

        self.settings_dir = os.path.join(
            "tests", "integrity", "bpsk_qam", "settings")
        self.global_output_dir = os.path.join(
            "tests", "integrity", "bpsk_qam", "results")

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)

    def test_one_sender_one_receiver_AWGN(self) -> None:
        self.settings_dir = os.path.join(
            self.settings_dir, "settings_oneTx_oneRx_AWGN")
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_oneTx_oneRx_AWGN")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", self.output_dir]
        hermes.hermes(self.arguments_hermes)

        results_simulation = loadmat(
            os.path.join(
                self.output_dir,
                "statistics.mat"))
        expected_results_simulation = loadmat(
            os.path.join(
                self.global_output_dir,
                "statistics_oneTx_oneRx_AWGN.mat")
        )
        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation))

    def test_threeTx_twoRx_AWGN(self) -> None:
        no_tx_modems = 3

        self.settings_dir = os.path.join(
            self.settings_dir, "settings_threeTx_twoRx_AWGN")
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_threeTx_twoRx_AWGN")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", self.output_dir]
        hermes.hermes(self.arguments_hermes)

        results_simulation = loadmat(
            os.path.join(
                self.output_dir,
                "statistics.mat"))
        expected_results_simulation = loadmat(
            os.path.join(
                self.global_output_dir,
                "statistics_threeTx_twoRx_AWGN.mat")
        )
        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation,
                no_tx_modems))

    def test_oneTx_oneRx_rayleighFading(self) -> None:

        self.settings_dir = os.path.join(
            self.settings_dir,
            "settings_oneTx_oneRx_rayleighFading")
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_oneTx_oneRx_rayleighFading")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", self.output_dir]
        hermes.hermes(self.arguments_hermes)

        results_simulation = loadmat(
            os.path.join(
                self.output_dir,
                "statistics.mat"))
        expected_results_simulation = loadmat(
            os.path.join(self.global_output_dir,
                         "statistics_oneTx_oneRx_rayleighFading.mat")
        )
        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation))
        self.assertTrue(compare_theory_simulation_errors(
            expected_results_simulation["ber_mean"],
            results_simulation["ber_upper"],
            results_simulation["ber_lower"]
        )
        )
        self.assertTrue(compare_theory_simulation_errors(
            expected_results_simulation["fer_mean"],
            results_simulation["fer_upper"],
            results_simulation["fer_lower"]
        )
        )
