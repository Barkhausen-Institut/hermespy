from enum import Enum
import numpy as np
import scipy.constants as const
from scipy import sin, cos
import matplotlib.pyplot as plt
from abc import abstractmethod


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
    __is_linear: bool           # Flag indicating a one-dimensional array

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
        self.__is_linear = False

        # Automatically detect linearity in default configurations, where all sensor elements
        # are oriented along the local x-axis.
        axis_sums = np.sum(self.__topology, axis=0)

        if (axis_sums[1] + axis_sums[2]) < 1e-10:
            self.__is_linear = True

    @abstractmethod
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
                A vector or matrix containing the computed beamforming weights.
                The first matrix dimension indicates the resulting number of streams after beamforming.
                The last dimension must be equal to the number of sensors within the considered array.
                Each column within the weight matrix may only contain one non-zero entry.
        """

        return np.identity(self.__topology.shape[0], dtype=complex)

    def gain(self, direction: TransmissionDirection, azimuth: float, elevation: float, weights: np.ndarray) -> complex:
        """Compute the complex gain coefficient towards a specific steering angle.

        The wave is assumed to originate from / impinge onto a point target in the arrays far-field.
        A small transmitted / received bandwidth compared to the array dimensions is assumed.

        Args:
            direction (TransmissionDirection):
                The direction of transmission, i.e. receive or transmit mode.

            azimuth (float):
                The considered azimuth angle in radians.

            elevation (float):
                The considered elevation angle in radians.

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

        wave_vector = self.wave_vector(azimuth, elevation)

        if direction == TransmissionDirection.Tx:
            wave_vector *= -1

        steering = np.array([np.exp(1j * wave_vector @ p) for p in self.__topology], dtype=complex)
        gain = weights @ steering

        return gain

    def wave_vector(self, azimuth: float, elevation: float) -> np.array:
        """Compute the three-dimensional wave vector of a far-field wave depending on arrival angles.

        A wave vector describes the phase of a planar wave depending on the considered position in space.

        Args:
            azimuth (float):
                Azimuth arrival angle in radians.

            elevation (float):
                Elevation angle in radians.
                For linear arrays the elevation can be assumed zero since the component does not apply.

        Returns:
            np.array:
                A three-dimensional wave vector in radians.
        """

        return 2 * const.pi * self.center_frequency / const.c * np.array([sin(azimuth) * cos(elevation),
                                                                          sin(elevation),
                                                                          cos(azimuth) * cos(elevation)])

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
                If the first dimension `topology` is smaller than 1 or its second dimension is larger than 3.
        """

        if len(topology.shape) > 2:
            raise ValueError("The topology array must be of dimension 2")

        if topology.shape[0] < 1:
            raise ValueError("The topology must contain at least one sensor")

        if len(topology.shape) > 1:

            if topology.shape[1] > 3:
                raise ValueError("The second topology dimension must contain 3 fields (xyz)")

            self.__topology = np.zeros((topology.shape[0], 3), dtype=float)
            self.__topology[:, :topology.shape[1]] = topology

        else:

            self.__topology = np.zeros((topology.shape[0], 3), dtype=float)
            self.__topology[:, 0] = topology

    @property
    @abstractmethod
    def num_streams(self) -> int:
        """The number of streams available after beamforming.

        Returns (int):
            The number of available streams.
        """

        # Standard beamforming applications compress multiple antenna signals into a single one.
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

    @property
    def is_linear(self) -> bool:
        """Access the configured linearity flag.

        Returns:
            bool:
                A boolean flag indicating whether this array is considered to be one-dimensional.
        """

        return self.__is_linear

    @is_linear.setter
    def is_linear(self, is_linear: bool) -> None:
        """Modify the configured linearity flag.

        Args:
            is_linear (bool):
                A boolean flag indicating whether this array is considered to be one-dimensional.
        """

        self.__is_linear = is_linear

    def inspect(self, weights: np.ndarray,
                azimuth_candidates: np.array = None,
                elevation_candidates: np.array = None,
                direction: TransmissionDirection = TransmissionDirection.Tx,
                normalized: bool = True) -> None:
        """Display the beamformer's spatial power pattern.

        Args:
            weights (np.ndarray):
                The beamforming weights for which to render the power pattern.

            azimuth_candidates (np.array, optional):
                An array of azimuth candidates to be sampled into the rendering.

            elevation_candidates (np.array, optional):
                An array of elevation candidates to be sampled into the rendering.

            direction (TransmissionDirection, optional):
                The direction of transmission, i.e. receive or transmit mode.
                By default transmit mode is assumed.

            normalized (bool, optional):
                Normalize the visualized gains.
                Enabled by default.

        Raises:
            ValueError:
                Should the number of `weights` not match the configured topology.
        """

        if weights.shape[-1] != self.__topology.shape[0]:
            raise ValueError("The number of beamforming weights must match the number of antennas")

        # Make weights vectors matrices with a single row to simplify following routines
        if len(weights.shape) == 1:
            weights = weights[np.newaxis, ...]

        # By default, the candidates are sampled in 1 degree steps over an 180 degree field of view
        if azimuth_candidates is None:

            if elevation_candidates is None:

                if self.is_linear:

                    azimuth_candidates = .5 * np.linspace(-const.pi, const.pi, 180, dtype=float)

                else:

                    azimuth_candidates = .5 * np.linspace(-const.pi, const.pi,  60, dtype=float)
                    elevation_candidates = .5 * np.linspace(-const.pi, const.pi, 60, dtype=float)

            else:

                raise ValueError("Defining elevation candidates without azimuth is currently not supported")

        if azimuth_candidates.shape[0] < 1:
            raise ValueError("Candidates must contain at least one angle")

        figure, axes = plt.subplots(weights.shape[0])
        figure.suptitle("Beamforming Inspection")

        # 1-D visualization mode
        if elevation_candidates is None:

            graph = np.empty((weights.shape[0], azimuth_candidates.shape[0]), dtype=float)
            for c, candidate in enumerate(azimuth_candidates):

                graph[:, c] = np.absolute(self.gain(direction, candidate, 0.0, weights))

            if normalized:
                graph /= self.__topology.shape[0]

            if weights.shape[0] > 1:

                for s in range(weights.shape[0]):

                    axes[s].set_title("Subarray #{}".format(s))
                    axes[s].plot(azimuth_candidates, graph[s, :])
                    axes[s].set(xlabel="Azimuth", ylabel="Magnitude")

            else:

                axes.plot(azimuth_candidates, graph[0, :])
                axes.set(xlabel="Azimuth", ylabel="Magnitude")

        # 2-D visualization mode
        else:

            graph = np.empty((weights.shape[0], azimuth_candidates.shape[0], elevation_candidates.shape[0]), dtype=float)
            for index_azimuth, azimuth in enumerate(azimuth_candidates):
                for index_elevation, elevation in enumerate(elevation_candidates):

                    graph[:, index_azimuth, index_elevation] = np.absolute(
                        self.gain(direction, azimuth, elevation, weights))

            if normalized:
                graph /= self.__topology.shape[0]

            if weights.shape[0] > 1:

                for s in range(weights.shape[0]):

                    axes[s].set_title("Subarray #{}".format(s))
                    axes[s].imshow(graph[s, ::].T)
                    axes[s].set(xlabel="Azimuth", ylabel="Magnitude")

            else:

                axes.imshow(graph[0, ::].T)
                axes.set(xlabel="Azimuth", ylabel="Elevation")

