import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

from beamformer import Beamformer, TransmissionDirection


class ConventionalBeamformer(Beamformer):

    def __init__(self, topology: np.ndarray, center_frequency: float):
        """Class initialization.

        Args:
            topology (np.ndarray):
                A matrix of m x 3 entries describing the sensor array topology.
                Each row represents the xyz-location of a single antenna within an array of m antennas.

            center_frequency (float):
                The center frequency in Hz of the RF-signal to be steered.
        """

        Beamformer.__init__(self, topology, center_frequency)

    def weights(self, direction: TransmissionDirection, azimuth: float, elevation: float) -> np.array:
        """Compute the beamforming weights towards a desired direction.

        Args:
            direction (TransmissionDirection):
                The direction of transmission, i.e. receive or transmit mode.

            azimuth (float):
                The considered azimuth angle in radians.

            elevation (float):
                The considered elevation angle in radians.

        Returns:
            np.array:
                A vector containing the computed beamforming weights.
        """

        wave_vector = self.wave_vector(azimuth, elevation)

        if direction == TransmissionDirection.Rx:
            wave_vector *= -1

        return np.array([np.exp(1j * wave_vector @ p) for p in self.topology], dtype=complex)

    @property
    def num_streams(self) -> int:
        """The number of channels available streams after beamforming.

        Returns (int):
            The number of available streams.
        """

        return 1
