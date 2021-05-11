import configparser
import numpy as np


class ParametersRfChain:
    """This class implements the parser of the RF-chain parameters."""

    supported_power_amplifier_models = ["NONE", "CLIP", "RAPP", "SALEH", "CUSTOM"]

    def __init__(self) -> None:

        # RF Chain parameters
        self.power_amplifier = "NONE"
        self.input_backoff_pa_db = 0.
        self.rapp_smoothness_factor = 1.
        self.saleh_alpha_a = 2.
        self.saleh_alpha_phi = 1.
        self.saleh_beta_a = 2.
        self.saleh_beta_phi = 1.
        self.custom_pa_input = np.array([])
        self.custom_pa_output = np.array([])
        self.custom_pa_gain = np.array([])
        self.custom_pa_phase = np.array([])
        self.adjust_power_after_pa = False

    def read_params(self, file_name: str) -> None:
        """This method reads and checks the validity of all the parameters from the file 'file_name'."""
        config = configparser.ConfigParser()
        config.read(file_name)

        section = config['PowerAmplifier']

        self.power_amplifier = section.get("power_amplifier", fallback="NONE").upper()
        self.input_backoff_pa_db = section.getfloat("input_backoff_dB", fallback=0)
        self.rapp_smoothness_factor = section.getfloat("rapp_smoothness_factor")
        self.saleh_alpha_a = section.getfloat("saleh_alpha_a")
        self.saleh_alpha_phi = section.getfloat("saleh_alpha_phi")
        self.saleh_beta_a = section.getfloat("saleh_beta_a")
        self.saleh_beta_phi = section.getfloat("saleh_beta_phi")
        custom_pa_input = section.get("custom_pa_input")
        if custom_pa_input:
            self.custom_pa_input = np.fromstring(custom_pa_input, dtype=float, sep=',')
        custom_pa_output = section.get("custom_pa_output")
        if custom_pa_output:
            self.custom_pa_output = np.fromstring(custom_pa_output, dtype=float, sep=',')
        custom_pa_phase = section.get("custom_pa_phase")
        if custom_pa_phase:
            self.custom_pa_phase = np.fromstring(custom_pa_phase, dtype=float, sep=',')

        self.adjust_power_after_pa = section.getboolean("adjust_power_after_pa")

        self._check_params()

    def _check_params(self) -> None:

        if self.power_amplifier not in ParametersRfChain.supported_power_amplifier_models:
            raise ValueError(f'power amplifier model {self.power_amplifier} not supported')

        if self.power_amplifier == "RAPP" and self.rapp_smoothness_factor <= 0:
            raise ValueError(f"Smoothness factor ({self.rapp_smoothness_factor}) in Rapp's model must be > 0")

        if self.power_amplifier == "CUSTOM":
            if (self.custom_pa_input.shape != self.custom_pa_output.shape or
                    self.custom_pa_input.shape != self.custom_pa_phase.shape):
                raise ValueError('Vectors for custom power amplifier must have the same length')

            if np.any(np.diff(self.custom_pa_input) <= 0) or np.any(self.custom_pa_input < 0):
                raise ValueError(f"{self.custom_pa_input} must be a non-negative increasing vector")

            if self.custom_pa_input[0] == 0:
                self.custom_pa_gain = self.custom_pa_output[1:] / self.custom_pa_input[1:]
            else:
                self.custom_pa_gain = self.custom_pa_output / self.custom_pa_input
                self.custom_pa_input = np.insert(self.custom_pa_input, 0, 0)
                self.custom_pa_output = np.insert(self.custom_pa_output, 0, 0)
                self.custom_pa_phase = np.insert(self.custom_pa_phase, 0, 0)

            self.custom_pa_gain = np.insert(self.custom_pa_gain, 0, 1)
