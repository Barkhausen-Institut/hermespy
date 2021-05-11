import unittest
import os
import shutil

import numpy as np
from scipy.io import loadmat

import hermes
from tests.integrity.utils import compare_mat_files


class TestOfdm(unittest.TestCase):
    def setUp(self) -> None:
        # the path is relative to os.getcwd(), i.e. sys.path()

        self.settings_dir = os.path.join(
            "tests", "integrity", "ofdm", "settings")
        self.global_output_dir = os.path.join(
            "tests", "integrity", "ofdm", "results")

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

    def test_5gPhyChannelModel_TDLA_oneTx_oneRx_SISO(self) -> None:
        """ It is expected that the BER  will be higher with a higher rms delay spread, 
        because part of the impulse response will have a delay higher than 
        the OFDM guard interval.Anyway, it 's more of a sanity check than a unit test."""

        self.settings_dir = os.path.join(
            self.settings_dir,
            "settings_oneTx_oneRx_5gChannelModel_TDLA_SISO",
            "RMS30"
        )
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_oneTx_oneRx_5gChannelModel_TDLA_SISO")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", os.path.join(self.output_dir, "RMS30")]
        hermes.hermes(self.arguments_hermes)
        results_simulation_rms30 = loadmat(
            os.path.join(
                self.output_dir,
                "RMS30",
                "statistics.mat"))

        self.settings_dir = os.path.join(
            self.settings_dir,
            "..",
            "RMS90"
        )
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_oneTx_oneRx_5gChannelModel_TDLA_SISO")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", os.path.join(self.output_dir, "RMS90")]
        hermes.hermes(self.arguments_hermes)
        results_simulation_rms90 = loadmat(
            os.path.join(
                self.output_dir,
                "RMS90",
                "statistics.mat"))

        self.assertTrue(np.all(
            results_simulation_rms30["ber_mean"] < results_simulation_rms90["ber_mean"]
            )
        )

    def test_one_sender_one_receiver_AWGN_precoding(self) -> None:
        self.settings_dir = os.path.join(
            self.settings_dir,
            "settings_oneTx_oneRx_AWGN_precoding")
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_oneTx_oneRx_AWGN_precoding")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", self.output_dir]
        hermes.hermes(self.arguments_hermes)

        results_simulation = loadmat(
            os.path.join(
                self.output_dir,
                "statistics.mat"))
        expected_results_simulation = loadmat(
            os.path.join(self.global_output_dir,
                         "statistics_oneTx_oneRx_AWGN_precoding.mat")
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

    def test_twoTx_oneRx_OFDM_QAM_AWGN(self) -> None:
        no_tx_modems = 2

        self.settings_dir = os.path.join(
            self.settings_dir, "settings_twoTx_oneRx_OFDM_QAM_AWGN")
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_twoTx_oneRx_OFDM_QAM_AWGN")

        self.arguments_hermes = [
            "-p", self.settings_dir, "-o", self.output_dir]
        hermes.hermes(self.arguments_hermes)

        results_simulation = loadmat(
            os.path.join(
                self.output_dir,
                "statistics.mat"))
        expected_results_simulation = loadmat(
            os.path.join(self.global_output_dir,
                         "statistics_twoTx_oneRx_OFDM_QAM_AWGN.mat")
        )
        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation,
                no_tx_modems))

    def test_oneTx_oneRx_Multipath(self) -> None:
        no_tx_modems = 1

        self.settings_dir = os.path.join(
            self.settings_dir, "settings_oneTx_oneRx_Multipath")
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_oneTx_oneRx_Multipath")

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
                "statistics_oneTx_oneRx_Multipath.mat")
        )
        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation,
                no_tx_modems))

    def test_threeTx_twoRx_Multipath(self) -> None:
        no_tx_modems = 1

        self.settings_dir = os.path.join(
            self.settings_dir, "settings_threeTx_twoRx_Multipath")
        self.output_dir = os.path.join(
            self.global_output_dir,
            "results_threeTx_twoRx_Multipath")

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
                "statistics_threeTx_twoRx_Multipath.mat")
        )
        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation,
                no_tx_modems))
