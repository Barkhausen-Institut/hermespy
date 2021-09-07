from enum import Enum
import numpy as np
import scipy.constants as const
from scipy import sin, cos


class TransmissionDirection(Enum):
    """Direction of transmission.

    Required information for beam-forming mode.
    """

    Rx = 1
    Tx = 2


class Beamformer:
    """Base class for antenna array steering weight calculation.

    Caution: Beamforming is only applicable to spatial system models.
    """

    __topology: np.ndarray      # Antenna array topology, implicitly contains the number of antennas
    __center_frequency: float   # Center frequency of the considered RF-signal

    def __init__(self, topology: np.ndarray, center_frequency: float):
        """Class initialization.

        Args:
            topology (np.ndarray):
                A matrix of m x 3 entries describing the sensor array topology.
                Each row represents the xyz-location of a single antenna within an array of m antennas.

            center_frequency (float):
                The center frequency in Hz of the RF-signal to be steered.
        """

        self.topology = topology
        self.center_frequency = center_frequency

    def weights(self, ) -> np.array:

        return np.identity(self.__topology.shape[0], dtype=complex)

    def gain(self, direction: TransmissionDirection, elevation: float, azimuth: float, weights: np.ndarray) -> complex:
        """Compute the complex gain coefficient towards a specific steering angle.

        The wave is assumed to originate from / impinge onto a point target in the arrays far-field.
        A small transmitted / received bandwidth compared to the array dimensions is assumed.

        Args:
            direction (TransmissionDirection):
                The direction of transmission, i.e. receive or transmit mode.

            elevation (float):
                The considered elevation angle in radians.

            azimuth (float):
                The considered azimuth angle in radians.

            weights (np.ndarray):
                The selected beamforming weights.

        Returns:
            complex:
                The complex gain towards the considered angles given the selected `weights`.
                The gain is not normalized, normalization requires a division by the number of antennas.

        Raises:
            ValueError:
                Should the number of `weights` not match the configured topology.
        """

        if weights.shape[-1] != self.__topology.shape[0]:
            raise ValueError("The number of beamforming weights must match the number of antennas")

        # Wave vector
        k = -2 * const.pi * self.center_frequency / const.c * np.array([sin(elevation) * cos(azimuth),
                                                                        sin(elevation) * sin(azimuth),
                                                                        cos(elevation)])
        wave = np.array([np.exp(-1j * k @ p) for p in self.__topology], dtype=complex)

        gain = weights @ wave
        return gain

    @property
    def topology(self) -> np.ndarray:
        """Access the configured sensor array topology.

        Returns:
            np.ndarray:
                A matrix of m x 3 entries describing the sensor array topology.
                Each row represents the xyz-location of a single antenna within an array of m antennas.
        """

        return self.__topology

    @topology.setter
    def topology(self, topology: np.ndarray) -> None:
        """Update the configured sensor array topology.

        Args:
            topology (np.ndarray):
                A matrix of m x 3 entries describing the sensor array topology.
                Each row represents the xyz-location of a single antenna within an array of m antennas.

        Raises:
            ValueError:
                If the first `topology` is smaller than 1 or its second dimension is not 3.
        """

        if len(topology.shape) != 2:
            raise ValueError("The topology array must be of dimension 2")

        if topology.shape[0] < 1:
            raise ValueError("The topology must contain at least one sensor")

        if topology.shape[1] != 3:
            raise ValueError("The second topology dimension must contain 3 fields (xyz)")

        self.__topology = topology

    @property
    def num_channels(self) -> int:
        """The number of channels available after beamforming.

        Returns (int):
            The number of available channels.
        """

        # Standard beamforming applications compress all antenna signals into a single one.
        # However, since the default beamformer does no beamforming at all, it returns the number of antennas.
        return self.__topology.shape[0]

    @property
    def center_frequency(self) -> float:
        """Access the configured center frequency of the steered RF-signal.

        Returns:
            float:
                Center frequency in Hz.
        """

        return self.__center_frequency

    @center_frequency.setter
    def center_frequency(self, center_frequency: float) -> None:
        """Modify the configured center frequency of the steered RF-signal.

        Args:
            center_frequency (float):
                Center frequency in Hz.

        Raises:
            ValueError:
                If center frequency is less or equal to zero.
        """

        if center_frequency <= 0:
            raise ValueError("Center frequency must be greater than zero")

        self.__center_frequency = center_frequency

    @property
    def center_wavelength(self) -> float:
        """Access the configured center wavelength of the steered RF-signal.

        Returns:
            float:
                Center wavelength in m.
        """

        return const.c / self.__center_frequency
