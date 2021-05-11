import unittest
import os
import shutil

from scipy.io import loadmat

from tests.integrity.utils import compare_mat_files
from hermes import hermes


class TestExamples(unittest.TestCase):
    def setUp(self) -> None:
        self.settings_dir = os.path.join(
            "tests", "integrity", "examples", "_settings"
        )
        self.global_output_dir = os.path.join(
            "tests", "integrity", "examples", "results"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)

    def test_chirp_qam(self) -> None:
        self.settings_dir = os.path.join(
            self.settings_dir, "chirp_qam"
        )
        self.output_dir = os.path.join(
            self.global_output_dir, "temp_results_chirp_qam"
        )

        self.arguments_hermes = ["-p", self.settings_dir, "-o", self.output_dir]

        hermes(self.arguments_hermes)
        expected_results_simulation = loadmat(
            os.path.join(self.global_output_dir, "statistics_chirp_qam.mat"))
        results_simulation = loadmat(
            os.path.join(self.output_dir, "statistics.mat")
        )

        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation))

    def test_chirp_fsk_lora(self) -> None:
        self.settings_dir = os.path.join(
            self.settings_dir, "chirp_fsk_lora"
        )
        self.output_dir = os.path.join(
            self.global_output_dir, "temp_chirp_fsk_lora"
        )

        self.arguments_hermes = ["-p", self.settings_dir, "-o", self.output_dir]

        hermes(self.arguments_hermes)
        expected_results_simulation = loadmat(
            os.path.join(self.global_output_dir, "statistics_chirp_fsk_lora.mat"))
        results_simulation = loadmat(
            os.path.join(self.output_dir, "statistics.mat")
        )

        self.assertTrue(
            compare_mat_files(
                results_simulation,
                expected_results_simulation))