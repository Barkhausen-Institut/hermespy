import numpy as np

from parameters_parser.parameters_rf_chain import ParametersRfChain


class PowerAmplifier:
    """This module implements a distortion model for a power amplifier (PA).

    The following PA models are currently supported:

    - Rapp's model, as described in
      C. Rapp, "Effects of HPA Nonlinearity on a 4-PI DPSK/OFDM Signal for a digital Sound Broadcasting System"
      in Proc. Eur. Conf. Satellite Comm., 1991

    - Saleh's model, as described in
      A.A.M. Saleh, "Frequency-independent and frequency dependent nonlinear models of TWT amplifiers"
      in IEEE Trans. Comm., Nov. 1981

    - any custom AM/AM and AM/PM response as a vector

    Currently only memoryless models are supported.
    """

    def __init__(self, params: ParametersRfChain, tx_power: float):
        """Creates a power amplifier object

        Args:
            params(ParametersRfChain): object containing all the RF-chain parameters
            tx_power(float): average power of transmitted signal (in linear scale)
        """
        self.model = params.power_amplifier
        self.params = params

        self.saturation_amplitude: float
        if self.model is not None and self.model != "NONE":
            self.power_backoff = 10**(params.input_backoff_pa_db/10)

            saturation_power = tx_power * self.power_backoff
            self.saturation_amplitude = np.sqrt(saturation_power)

    def send(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of the input signal according to the desired model

        Args:
            input_signal(np.ndarray): vector with input signal

        Returns:
            numpy.ndarray: Distorted signal of same size as `input_signal`.
        """

        output_signal = np.ndarray([])
        if self.model == "NONE":
            output_signal = input_signal

        elif self.model == "CLIP":
            # clipping
            clip_idx = np.nonzero(np.abs(input_signal) > self.saturation_amplitude)
            output_signal = np.copy(input_signal)
            output_signal[clip_idx] = self.saturation_amplitude * np.exp(1j * np.angle(input_signal[clip_idx]))

        elif self.model == "RAPP":
            p = self.params.rapp_smoothness_factor
            gain = (1 + (np.abs(input_signal) / self.saturation_amplitude)**(2 * p))**(-1/2/p)
            output_signal = input_signal * gain

        elif self.model == "SALEH":
            amp = np.abs(input_signal) / self.saturation_amplitude
            gain = self.params.saleh_alpha_a / (1 + self.params.saleh_beta_a * amp**2)
            phase_shift = self.params.saleh_alpha_phi * amp**2 / (1 + self.params.saleh_beta_phi * amp**2)
            output_signal = input_signal * gain * np.exp(1j * phase_shift)

        elif self.model == "CUSTOM":
            amp = np.abs(input_signal) / self.saturation_amplitude
            gain = np.interp(amp, self.params.custom_pa_input, self.params.custom_pa_gain)
            phase_shift = np.interp(amp, self.params.custom_pa_input, self.params.custom_pa_phase)
            output_signal = input_signal * gain * np.exp(1j * phase_shift)

        else:
            ValueError(f"Power amplifier model {self.model} not supported")

        if self.params.adjust_power_after_pa:
            loss = np.linalg.norm(output_signal) / np.linalg.norm(input_signal)
            output_signal = output_signal / loss

        return output_signal
