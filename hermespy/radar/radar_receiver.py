from __future__ import annotations
from ruamel.yaml import RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap
from typing import Type, List, Optional
import numpy as np
import numpy.random as rnd

from source import BitsSource
from modem.receiver import Receiver
from modem.waveform_generator import WaveformGenerator
from simulator_core.statistics.radar_output import RadarOutput


class RadarReceiver(Receiver):

    yaml_tag = 'RadarReceiver'

    def __init__(self, **kwargs) -> None:
        Receiver.__init__(self, **kwargs)

    def receive(self, input_signal: np.ndarray, noise_var: float) -> np.ndarray:
        """Demodulates the signal received.

        The received signal may be distorted by RF imperfections before demodulation and decoding.

        Args:
            input_signal (np.ndarray): Received signal.
            noise_var (float): noise variance (for equalization).

        Returns:
            np.array: Detected bits as a list of data blocks for the drop.
        """
        rx_signal = self.rf_chain.receive(input_signal)

        # If no receiving waveform generator is configured, no signal is being received
        if self.waveform_generator is None:
            return np.empty(0, dtype=complex)

        # normalize signal to expected input power
        rx_signal = rx_signal / np.sqrt(1.0)  # TODO: Re-implement pair power factor
        noise_var = noise_var / 1.0  # TODO: Re-implement pair power factor

        radar_output = RadarOutput()
        timestamp_in_samples = 0

        while rx_signal.size:
            initial_size = rx_signal.shape[1]
            frame_output, rx_signal = self.waveform_generator.receive_frame(
                rx_signal, timestamp_in_samples, noise_var)

            if rx_signal.size:
                timestamp_in_samples += initial_size - rx_signal.shape[1]

            radar_output = radar_output + frame_output

        return radar_output
