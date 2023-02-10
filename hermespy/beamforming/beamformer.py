# -*- coding: utf-8 -*-
"""
==========
Beamformer
==========

Beamforming is split into the prototype classes :class:`.TransmitBeamformer` and :class:`.ReceiveBeamformer`
for beamforming operations during signal transmission and reception, respectively.
They are both derived from the base :class:`BeamformerBase`.
This is due to the fact that some beamforming algorithms may be exclusive to transmission or reception use-cases.
Should a beamformer be applicable during both transmission and reception both prototypes can be inherited.
An example for such an implementation is the :class:`Conventional <.conventional.ConventionalBeamformer>` beamformer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from typing import Generic, Optional, Tuple, TypeVar, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.constants import pi

from hermespy.core import Executable, FloatingError, IdealAntenna, Operator, Receiver, Reception, SerializableEnum, Signal, Transmitter, UniformArray
from hermespy.precoding import ReceiveStreamDecoder, TransmitStreamEncoder
from hermespy.precoding.precoding import Precoding
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FocusMode(SerializableEnum):
    """The focus mode of the beamformer."""

    SPHERICAL = 0
    """Focus points in spherical coordinates, i.e. azimuth and zenith angles in radians."""

    HORIZONTAL = 1
    """Focus points in horizontal coordinates, i.e. azimuth and elevation angles in radians."""

    CARTESIAN = 2
    """Focus points in Cartesian coordinates, i.e. xyz in m."""

    DEVICE = 3
    """Focus points considering peer devices."""


OT = TypeVar("OT", bound=Operator)
"""Type of operator."""


class BeamformerBase(ABC, Generic[OT]):
    """Base class for all beam steering precodings."""

    # Reference to the operator the beamformer is attached to
    __operator: OT | None

    def __init__(self, operator: OT | None = None) -> None:
        """Args:

        operator (OT, optional):
            The operator this beamformer is attached to.
            By default, the beamformer is considered floating.
        """

        self.operator = operator

    @property
    def operator(self) -> OT | None:
        """The operator this beamformer is assigned to.

        Returns:

            Handle to the operator.
            `None` if the beamformer is considered floating.
        """

        return self.__operator

    @operator.setter
    def operator(self, value: OT | None) -> None:

        self.__operator = value


class TransmitBeamformer(BeamformerBase[Transmitter], TransmitStreamEncoder, ABC):
    """Base class for beam steering precodings during signal transmissions."""

    __focus_points: np.ndarray
    __focus_mode: FocusMode

    def __init__(self, operator: Transmitter | None = None) -> None:
        """Args:

        operator (Transmitter, optional):
            The operator this beamformer is attached to.
            By default, the beamformer is considered floating.
        """

        self.__focus_points = np.array([[0.0, 0.0]], dtype=float)
        self.__focus_mode = FocusMode.SPHERICAL

        BeamformerBase.__init__(self, operator=operator)

    @abstractproperty
    def num_transmit_input_streams(self) -> int:
        """Number of input streams required by this beamformer.

        Returns:

            Number of input streams.
        """
        ...  # pragma no cover

    @abstractproperty
    def num_transmit_output_streams(self) -> int:
        """Number of output streams generated by this beamformer.

        Returns:

            Number of output streams.
        """
        ...  # pragma no cover

    @abstractproperty
    def num_transmit_focus_angles(self) -> int:
        """Number of required transmit focus angles.

        Returns:

            Number of focus angles.
        """
        ...  # pragma no cover

    def encode_streams(self, streams: Signal) -> Signal:

        if streams.num_streams != self.num_transmit_input_streams:
            raise ValueError("Stream encoding configuration invalid, number of provided streams don't match the beamformer requirements")

        return self.transmit(streams)

    @TransmitStreamEncoder.precoding.setter  # type: ignore
    def precoding(self, precoding: Precoding) -> None:

        self.operator = precoding.modem  # type: ignore
        TransmitStreamEncoder.precoding.fset(self, precoding)  # type: ignore

    @abstractmethod
    def _encode(self, samples: np.ndarray, carrier_frequency: float, focus_angles: np.ndarray) -> np.ndarray:
        """Encode signal streams for transmit beamforming.

        Args:

            samples (np.ndarray):
                Signal samples, first dimension being the number of transmit antennas, second the number of samples.

            carrier_frequency (float):
                The assumed carrier central frequency of the samples.

            focus_angles (np.ndarray):
                Focused angles of departure in radians.
                Two-dimensional numpy array with the first dimension representing the number of focus points
                and the second dimension of magnitude two being the azimuth and elevation angles, respectively.

            azimuth (float):
                Azimuth angle of departure in Radians.

            zenith (float):
                Zenith angle of departure in Radians.
        """
        ...  # pragma no cover

    @property
    def transmit_focus(self) -> Tuple[np.ndarray, FocusMode]:
        """Focus points of the beamformer during transmission.

        Returns:

            - Numpy array of focus points elevation and azimuth angles
            - Focus mode
        """

        return self.__focus_points, self.__focus_mode

    @transmit_focus.setter  # type: ignore
    def transmit_focus(self, value: Union[np.ndarray, Tuple[np.ndarray, FocusMode]]) -> None:

        if not isinstance(value, (tuple, list)):
            value = (value, self.__focus_mode)

        if value[0].ndim != 2:
            raise ValueError("Focus must be a two-dimensional array")

        if value[0].shape[0] != self.num_transmit_focus_angles:
            raise ValueError(f"Focus requires {self.num_transmit_focus_angles} points, but {value[0].shape[0]} were provided")

        self.__focus_points = value[0]
        self.__focus_mode = value[1]

    def transmit(self, signal: Signal, focus: Optional[np.ndarray] = None) -> Signal:
        """Focus a signal model towards a certain target.

        Args:

            signal (Signal):
                The signal to be steered.

            focus (np.ndarray, optional):
                Focus point of the steered signal power.

        Returns:

            Samples of the focused signal.
        """

        if self.operator is None:
            raise FloatingError("Unable to steer a signal over a floating beamformer")

        if self.operator.device is None:
            raise FloatingError("Unable to steer a signal over a floating operator")

        if signal.num_streams != self.num_transmit_input_streams:
            raise RuntimeError(f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_transmit_input_streams} streams")

        carrier_frequency = signal.carrier_frequency
        samples = signal.samples.copy()
        focus = self.transmit_focus[0] if focus is None else focus

        steered_samples = self._encode(samples, carrier_frequency, focus)
        return Signal(steered_samples, sampling_rate=signal.sampling_rate, carrier_frequency=signal.carrier_frequency)


class ReceiveBeamformer(BeamformerBase[Receiver], ReceiveStreamDecoder, ABC):
    """Base class for beam steering precodings during signal receptions.

    The beamformer is characterised by its required number of input streams :math:`N`,
    the resulting number of output streams :math:`M` and the supported number of focus points :math:`F`.
    Considering a beamformer is provided with a matrix of :math:`T` baseband samples
    :math:`\\mathbf{X} \\in \\mathbb{C}^{N \\times T}`, the beamforming process

    .. math::

       \\mathbf{Y} = \\mathcal{B}\\lbrace \\mathbf{X} \\rbrace \\quad \\text{with} \\quad \\mathbf{Y} \\in \\mathbb{C}^{M \\times T}

    can generally be described as a function compressing the number of streams while focusing the power towards
    the angles of interest :math:`F`.
    """

    __focus_points: np.ndarray
    __focus_mode: FocusMode

    def __init__(self, operator: Receiver | None = None) -> None:
        """Args:

        operator (Receiver, optional):
            The operator this beamformer is attached to.
            By default, the beamformer is considered floating.
        """

        self.__focus_points = np.array([[0.0, 0.0]], dtype=float)
        self.__focus_mode = FocusMode.SPHERICAL
        self.probe_focus_points = np.zeros((1, self.num_receive_focus_angles, 2), dtype=float)

        BeamformerBase.__init__(self, operator=operator)
        ReceiveStreamDecoder.__init__(self)

    @abstractproperty
    def num_receive_input_streams(self) -> int:
        """Number of input streams required by this beamformer.

        Dimension :math:`N` of the input sample matrix
        :math:`\\mathbf{X} \\in \\mathbb{C}^{N \\times T}`.

        Returns:

            Number of input streams :math:`N`.
        """
        ...  # pragma no cover

    @abstractproperty
    def num_receive_output_streams(self) -> int:
        """Number of output streams generated by this beamformer.

        Dimension :math:`M` of the output sample matrix
        :math:`\\mathbf{Y} \\in \\mathbb{C}^{M \\times T}`.

        Returns:

            Number of output streams :math:`M`.
        """
        ...  # pragma no cover

    @abstractproperty
    def num_receive_focus_angles(self) -> int:
        """Number of required receive focus angles.

        Returns:

            Number of focus angles :math:`F`.
        """
        ...  # pragma no cover

    def decode_streams(self, streams: Signal) -> Signal:

        if streams.num_streams != self.num_receive_input_streams:
            raise ValueError("Stream decoding configuration invalid, number of provided streams don't match the beamformer requirements")

        return self.receive(streams)

    @ReceiveStreamDecoder.precoding.setter  # type: ignore
    def precoding(self, precoding: Precoding) -> None:

        self.operator = precoding.modem  # type: ignore
        ReceiveStreamDecoder.precoding.fset(self, precoding)  # type: ignore

    @abstractmethod
    def _decode(self, samples: np.ndarray, carrier_frequency: float, angles: np.ndarray) -> np.ndarray:
        """Decode signal streams for receive beamforming.

        This method is called as a subroutine during :meth:`receive` and :meth:`probe`.

        Args:

            samples (np.ndarray):
                Signal samples, first dimension being the number of signal streams :math:`N`, second the number of samples :math:`T`.

            carrier_frequency (float):
                The assumed carrier central frequency of the samples :math:`f_\\mathrm{c}`.

            angles: (np.ndarray):
                Spherical coordinate system angles of arrival in radians.
                A three-dimensional numpy array with the first dimension representing the number of angles,
                the second dimension of magnitude number of focus points :math:`F`,
                and the third dimension containing the azimuth and zenith angle in radians, respectively.

        Returns:

            Stream samples of the focused signal towards all focus points.
            A three-dimensional numpy array with the first dimension representing the number of focus points,
            the second dimension the number of returned streams and the third dimension the amount of samples.
        """
        ...  # pragma no cover

    @property
    def receive_focus(self) -> Tuple[np.ndarray, FocusMode]:
        """Focus points of the beamformer during reception.

        Returns:

            - Numpy array of focus points elevation and azimuth angles
            - Focus mode
        """

        return self.__focus_points, self.__focus_mode

    @receive_focus.setter
    def receive_focus(self, value: Union[np.ndarray, Tuple[np.ndarray, FocusMode]]) -> None:

        if not isinstance(value, (tuple, list)):
            value = (value, self.__focus_mode)

        if value[0].ndim != 2:
            raise ValueError("Focus must be a two-dimensional array")

        if value[0].shape[0] != self.num_receive_focus_angles:
            raise ValueError(f"Focus requires {self.num_receive_focus_angles} points, but {value[0].shape[0]} were provided")

        self.__focus_points = value[0]
        self.__focus_mode = value[1]

    def receive(self, signal: Signal, focus_points: Optional[np.ndarray] = None, focus_mode: FocusMode = FocusMode.SPHERICAL) -> Signal:
        """Focus a signal model towards a certain target.

        Args:

            signal (Signal):
                The signal to be steered.

            focus_points (np.ndarray, optional):
                Focus point of the steered signal power.
                Two-dimensional numpy array with the first dimension representing the number of points
                and the second dimension representing the point values.

            focus_mode (FocusMode, optional):
                Type of focus points.
                By default, spherical coordinates are expected.

        Returns:

            Signal focused towards the requested focus points.

        Raises:

            FloatingError: If the operator or operator device are not yet specified.
            RuntimeError: If the number of signal streams does not match the number of required input streams.
        """

        if self.operator is None:
            raise FloatingError("Unable to steer a signal over a floating beamformer")

        if self.operator.device is None:
            raise FloatingError("Unable to steer a signal over a floating operator")

        if signal.num_streams != self.num_receive_input_streams:
            raise RuntimeError(f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_receive_input_streams} streams")

        carrier_frequency = signal.carrier_frequency
        samples = signal.samples.copy()
        focus_angles = self.receive_focus[0][np.newaxis, ::] if focus_points is None else focus_points[np.newaxis, ::]

        beamformed_samples = self._decode(samples, carrier_frequency, focus_angles)
        return Signal(beamformed_samples[0, ::], signal.sampling_rate)

    @property
    def probe_focus_points(self) -> np.ndarray:
        """Default beamformer focus points during probing.

        Returns:

            The focus points as a three-dimensional numpy array, with the first dimension
            representing the probe index, the second dimension the point and the third dimension of magnitude
            two the point azimuth and zenith, respectively.

        Raises:

            ValueError: On invalid arguments.
        """

        return self.__probe_focus_points

    @probe_focus_points.setter
    def probe_focus_points(self, value: np.ndarray) -> None:

        # Expand points by new dimension if only a single focus tuple was requested
        if value.ndim == 2:
            value = value[np.newaxis, ::]

        if value.ndim != 3:
            raise ValueError("Probing focus points must be a three-dimensional array")

        if value.shape[1] != self.num_receive_focus_angles:
            raise ValueError(f"Focus requires {self.num_receive_focus_angles} points, but {value.shape[1]} were provided")

        self.__probe_focus_points = value

    def probe(self, signal: Signal, focus_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Focus a signal model towards certain directions of interest.

        Args:

            signal (Signal):
                The signal to be steered.

            focus_points (np.ndarray, optional):
                Focus point of the steered signal power.
                Two-dimensional numpy array with the first dimension representing the number of points
                and the second dimension representing the point values.

        Returns:

            Stream samples of the focused signal towards all focus points.
            A three-dimensional numpy array with the first dimension representing the number of focus points,
            the second dimension the number of returned streams and the third dimension the amount of samples.
        """

        focus_points = self.probe_focus_points if focus_points is None else focus_points

        if self.operator is None:
            raise FloatingError("Unable to steer a signal over a floating beamformer")

        if self.operator.device is None:
            raise FloatingError("Unable to steer a signal over a floating operator")

        if signal.num_streams != self.num_receive_input_streams:
            raise RuntimeError(f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_receive_input_streams} streams")

        carrier_frequency = signal.carrier_frequency
        samples = signal.samples.copy()

        return self._decode(samples, carrier_frequency, focus_points)

    def plot_receive_pattern(self, signal: Optional[Signal] = None) -> plt.Figure:
        """Visualize the beamformer instance's receive characteristics.

        Args:

            signal (Signal, optional):
                The impinging signal model for which to plot the receive characteristics.
                By default, a point-source signal at :math:`(\\theta=0, \\phi=0)` will be generated.

        Returns:

            A handle to the created matplotlib figure.
        """

        if signal is None:

            samples = self.operator.device.antennas.spherical_response(self.operator.device.carrier_frequency, 0.0, 0.0)
            signal = Signal(samples[:, np.newaxis], 1.0, self.operator.device.carrier_frequency)

        zenith_angles = np.linspace(0, 0.5 * pi, 31)
        azimuth_angles = np.linspace(-pi, pi, 31)
        zenith_samples, azimuth_samples = np.meshgrid(zenith_angles[1:], azimuth_angles)
        aoi = np.append(np.array([azimuth_samples.flatten(), zenith_samples.flatten()]).T, np.zeros((1, 2)), axis=0)

        beamformed_samples = self._decode(signal.samples, signal.carrier_frequency, aoi[:, np.newaxis, :])
        received_power = np.linalg.norm(beamformed_samples, axis=(1, 2))

        surface = np.array([received_power * np.cos(aoi[:, 0]) * np.sin(aoi[:, 1]), received_power * np.sin(aoi[:, 0]) * np.sin(aoi[:, 1]), received_power * np.cos(aoi[:, 1])], dtype=float)

        with Executable.style_context():

            figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
            figure.suptitle(f"{type(self).__name__} Receive Characteristics")

            triangles = tri.Triangulation(aoi[:, 0], aoi[:, 1])
            cmap = plt.cm.ScalarMappable(norm=colors.Normalize(received_power.min(), received_power.max()), cmap="jet")

            axes.plot_trisurf(surface[0, :], surface[1, :], surface[2, :], triangles=triangles.triangles, cmap=cmap.cmap, norm=cmap.norm, linewidth=0.0)

            return figure

    @classmethod
    def PlotReceivePattern(cls, array_topology: Optional[Tuple[int, ...]] = None, signal: Optional[Signal] = None) -> plt.Figure:
        """Visualize the beamformer class' receive characteristics.

        Args:

            array_topology (Tuple[int, ...], optional):
                The sensor array topology.
                By default, an :math:`8 \\times 8` uniform mimo matrix is assumed.

            signal (Signal, optional):
                The impinging signal model for which to plot the receive characteristics.
                By default, a point-source signal at :math:`(\\theta=0, \\phi=0)` will be generated.

        Returns:

            A handle to the created matplotlib figure.
        """
        array_topology = (8, 8) if array_topology is None else array_topology

        device = SimulatedDevice()
        device.carrier_frequency = 1e9
        device.antennas = UniformArray(IdealAntenna(), 0.5 * device.wavelength, (8, 8))

        class ReceiverMock(Receiver[Reception], ABC):
            def _receive(self, *args, **kwargs) -> Reception:
                raise NotImplementedError()  # pragma: no cover

            @property
            def energy(self) -> float:
                return 1.0  # pragma no cover

            @property
            def sampling_rate(self) -> float:
                return 1.0  # pragma no cover

            @property
            def frame_duration(self) -> float:
                return 1.0

            def _noise_power(self, strength: float, _) -> float:
                return strength

        operator = ReceiverMock()
        operator.slot = device.receivers

        beamformer = cls(operator=operator)

        return beamformer.plot_receive_pattern(signal)
