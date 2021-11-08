from __future__ import annotations
from typing import Type
from ruamel.yaml import SafeRepresenter, SafeConstructor, Node
from enum import Enum
import numpy as np
import scipy.constants as const
from scipy import sin, cos
import matplotlib.pyplot as plt

from hermespy.modem.precoding import Precoding, Precoder


class TransmissionDirection(Enum):
    """Direction of transmission.

    Required information for beam-forming mode.
    """

    Rx = 1
    Tx = 2


class Beamformer(Precoder):
    """Base class for antenna array steering weight calculation.

    Caution: Beamforming is only applicable to spatial system models.
    """

    yaml_tag = 'Beamformer'
    __center_frequency: float   # Assumed center frequency of the steered RF signal.

    def __init__(self,
                 precoding: Precoding = None,
                 center_frequency: float = 0.0):
        """Class initialization.

        Args:
            precoding (Precoding, optional):
                The precoding configuration this beamformer belongs to.

            center_frequency (float, optional):
                The center frequency in Hz of the RF-signal to be steered.
        """

        Precoder.__init__(self, precoding)
        self.center_frequency = center_frequency

    @classmethod
    def to_yaml(cls: Type[Beamformer], representer: SafeRepresenter, node: Beamformer) -> Node:
        """Serialize a `Beamformer` to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Beamformer):
                The `Beamformer` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        return representer.represent_scalar(cls.yaml_tag, "")

    @classmethod
    def from_yaml(cls: Type[Beamformer], constructor: SafeConstructor, node: Node) -> Beamformer:
        """Recall a new `Beamformer` from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Beamformer` serialization.

        Returns:
            Beamformer:
                Newly created `Beamformer` instance.
        """

        return cls()

    def encode(self, output_stream: np.matrix) -> np.matrix:
        """Add beam-forming weights to a data stream before transmission.

        Args:
            output_stream (np.matrix):
                The data streams feeding into the `Beamformer` to be encoded.

        Returns:
            np.matrix:
                The weighted data streams.
                The first matrix dimension is the number of transmit antennas,
                the second dimension the number of symbols transmitted per antenna.
        """
        pass

    def decode(self, input_stream: np.matrix) -> np.matrix:
        """Add inverse beam-forming weights to a data stream after reception.

        Args:
            input_stream (np.matrix):
                The symbol streams feeding into the `Beamformer` to be decoded.
                The first matrix dimension is the number of symbols per sensor,
                the second dimension the number of antennas.

        Returns:
            np.matrix:
                The decoded data streams.
                The first matrix dimension is the number of streams,
                the second dimension the number of symbols.
        """
        pass

    def num_inputs(self) -> int:
        """The required number of input symbol streams for transmit beam-forming.

        The default count for most beam-formers is one.

        Returns:
            int:
                The number of symbol streams.
        """

        return 1

    def num_outputs(self) -> int:
        """The generated number of output symbol streams after receive beam-forming.

        The default count for most beam-formers is one.

        Returns:
            int:
                The number of symbol streams.
        """

        return 1

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

        if TransmissionDirection == TransmissionDirection.Tx:
            return np.identity(self.precoding.get_outputs(self), dtype=complex)

        elif TransmissionDirection == TransmissionDirection.Rx:
            return np.identity(self.precoding.get_inputs(self), dtype=complex)

        else:
            raise RuntimeError("Unknown transmission direction")

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

        if weights.shape[-1] != self.__modem.topology.shape[0]:
            raise ValueError("The number of beamforming weights must match the number of antennas")

        wave_vector = self.wave_vector(azimuth, elevation)

        if direction == TransmissionDirection.Tx:
            wave_vector *= -1

        steering = np.array([np.exp(1j * wave_vector @ p) for p in self.__modem.topology], dtype=complex)
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
    def num_streams(self) -> int:
        """The number of streams available after beamforming.

        Returns (int):
            The number of available streams.
        """

        # Standard beamforming applications compress multiple antenna signals into a single one.
        # However, since the default beamformer does no beamforming at all, it returns the number of antennas.
        return self.__modem.topology.shape[0]

    @property
    def center_frequency(self) -> float:
        """Access the configured center frequency of the steered RF-signal.

        Returns:
            float:
                Center frequency in Hz.
        """

        if self.__center_frequency <= 0.0:
            return self.modem.carrier_frequency

        return self.__center_frequency

    @center_frequency.setter
    def center_frequency(self, center_frequency: float) -> None:
        """Modify the configured center frequency of the steered RF-signal.

        Args:
            center_frequency (float):
                Center frequency in Hz.
                If `center_frequency` is zero, the beamformer will assume the modem's configured center frequency.

        Raises:
            ValueError:
                If center frequency is less than zero.
        """

        if center_frequency < 0.0:
            raise ValueError("Center frequency must be greater or equal to zero")

        self.__center_frequency = center_frequency

    @property
    def center_wavelength(self) -> float:
        """Access the configured center wavelength of the steered RF-signal.

        Returns:
            float:
                Center wavelength in m.
        """

        return const.c / self.__center_frequency

    def inspect(self, weights: np.ndarray,
                azimuth_candidates: np.array = None,
                elevation_candidates: np.array = None,
                num_samples: int = None,
                direction: TransmissionDirection = TransmissionDirection.Tx,
                normalized: bool = True,
                interpolation: str = 'spline36') -> None:
        """Display the beamformer's spatial power pattern.

        Args:
            weights (np.ndarray):
                The beamforming weights for which to render the power pattern.

            azimuth_candidates (np.array, optional):
                An array of azimuth candidates to be sampled into the rendering.
                Definition is mutually exclusive to the definition of `num_samples`.

            elevation_candidates (np.array, optional):
                An array of elevation candidates to be sampled into the rendering.
                Definition is mutually exclusive to the definition of `num_samples`.

            num_samples (int, optional):
                The number of samples by which the field of view is sampled into the rendering.
                Definition is mutually exclusive to the definition of `azimuth_candidates` or `elevation_candidates`.

            direction (TransmissionDirection, optional):
                The direction of transmission, i.e. receive or transmit mode.
                By default transmit mode is assumed.

            normalized (bool, optional):
                Normalize the visualized gains.
                Enabled by default.

            interpolation
                Matplotlib interpolation mode.
                Enabled by default to spline36, set to None to disable.

        Raises:
            ValueError:
                Should the number of `weights` not match the configured topology.

            ValueError:
                Should both `num_samples` and `azimuth_candidates` or `elevation_candidates` be defined.
        """

        if weights.shape[-1] != self.__modem.topology.shape[0]:
            raise ValueError("The number of beamforming weights must match the number of antennas")

        # Make weights vectors matrices with a single row to simplify following routines
        if len(weights.shape) == 1:
            weights = weights[np.newaxis, ...]

        if num_samples is None:

            if self.__modem.linear_topology:
                num_samples = 180

            else:
                num_samples = 80

        elif azimuth_candidates or elevation_candidates is not None:
            raise ValueError("There can't be a definition of both sample count and samples")

        if num_samples < 1:
            raise ValueError("The number of rendered samples must be greater than zero")

        # By default, the candidates are sampled in 1 degree steps over an 180 degree field of view
        if azimuth_candidates is None:

            if elevation_candidates is None:

                if self.__modem.linear_topology:

                    azimuth_candidates = .5 * np.linspace(-const.pi, const.pi, num_samples, dtype=float)

                else:

                    azimuth_candidates = .5 * np.linspace(-const.pi, const.pi,  num_samples, dtype=float)
                    elevation_candidates = .5 * np.linspace(const.pi, -const.pi, num_samples, dtype=float)

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
                graph /= self.__modem.topology.shape[0]

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

            graph = np.empty((weights.shape[0], azimuth_candidates.shape[0], elevation_candidates.shape[0]),
                             dtype=float)
            for index_azimuth, azimuth in enumerate(azimuth_candidates):
                for index_elevation, elevation in enumerate(elevation_candidates):

                    graph[:, index_elevation, index_azimuth] = np.absolute(
                        self.gain(direction, azimuth, elevation, weights))

            if normalized:
                graph /= self.__modem.topology.shape[0]

            if weights.shape[0] > 1:

                for s in range(weights.shape[0]):

                    axes[s].set_title("Subarray #{}".format(s))
                    axes[s].imshow(graph[s, ::], interpolation=interpolation)
                    axes[s].set(xlabel="Azimuth", ylabel="Magnitude")

            else:

                axes.imshow(graph[0, ::], interpolation=interpolation)
                axes.set(xlabel="Azimuth", ylabel="Elevation")
