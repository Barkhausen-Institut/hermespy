import numpy as np
from numpy import random as rnd

from parameters_parser.parameters_rf_chain import ParametersRfChain
from modem.rf_chain_models.power_amplifier import PowerAmplifier


class RfChain:
    """Implements an RF chain model.

    Only PA is modelled.
    """

    def __init__(self, param: ParametersRfChain = None,
                 tx_power: float = 1.0) -> None:
        self.param = param
        if self.param is not None:
            self.power_amplifier = PowerAmplifier(self.param, tx_power)

    def send(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of signal in "input_signal".

        According to transmission impairments.
        """
        if self.param is not None:
            output_signal = self.power_amplifier.send(input_signal)
        else:
            output_signal = input_signal
        return output_signal

    def receive(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of signal in "input_signal".

        According to reception impairments.
        """
        return input_signal
