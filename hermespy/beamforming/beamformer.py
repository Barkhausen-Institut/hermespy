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
from typing import Generic, Literal, overload, Sequence, TypeVar

import numpy as np

from hermespy.core import (
    AntennaArray,
    Device,
    Direction,
    FloatingError,
    Operator,
    Receiver,
    Serializable,
    Signal,
    Transmitter,
)
from hermespy.precoding import Precoding, ReceiveStreamDecoder, TransmitStreamEncoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


FT = TypeVar("FT", bound="BeamFocus")
"""Type of beam focus."""


class BeamFocus(ABC, Serializable):
    """Single focus point of a beamformer."""

    property_blacklist = {"beamformer", "spherical_angles"}

    __beamformer: BeamformerBase | None  # Beamformer this focus point is assigned to

    def __init__(self) -> None:
        self.__beamformer = None

    @property
    @abstractmethod
    def spherical_angles(self) -> np.ndarray:
        """Azimuth and zenith angles in radians, towards which the beam is focused in spherical coordinates."""
        ...  # pragma: no cover

    @abstractmethod
    def copy(self: FT) -> FT:
        """Create a copy of this focus point.

        Returns:

            A copy of this focus point.
        """
        ...  # pragma: no cover

    @property
    def beamformer(self) -> BeamformerBase | None:
        """Beamformer this focus point is assigned to."""

        return self.__beamformer

    @beamformer.setter
    def beamformer(self, value: BeamformerBase | None) -> None:
        self.__beamformer = value

    def __str__(self) -> str:
        angles = self.spherical_angles
        return f"{self.__class__.__name__}({angles[0]:.2f}, {angles[1]:.2f})"


class DeviceFocus(BeamFocus):
    """Focus point targeting a device."""

    yaml_tag = "DeviceFocus"
    __device: Device  # Device focused by the beamformer

    def __init__(self, device: Device) -> None:
        """
        Args:

            device (Device): Device focused by the beamformer.
        """

        # Initialize base class
        BeamFocus.__init__(self)

        # Initialize class members
        self.__device = device

    def copy(self) -> DeviceFocus:
        return DeviceFocus(self.__device)

    @property
    def device(self) -> Device:
        """Device focused by the beamformer."""

        return self.__device

    @property
    def spherical_angles(self) -> np.ndarray:
        if self.beamformer is None:
            raise RuntimeError("Device focus requires the beamformer to be specified")

        if self.beamformer.operator is None:
            raise RuntimeError("Device focues requires the beaformer to be assigned to an operator")

        base = self.beamformer.operator.device
        if base is None:
            raise RuntimeError(
                "Device focus requires the beamformer's operator to be assigned to a device"
            )

        direction = Direction.From_Cartesian(
            self.__device.global_position - base.global_position, True
        )
        return direction.to_spherical()


class SphericalFocus(BeamFocus):
    """Focus point in spherical coordinates."""

    yaml_tag = "SphericalFocus"
    __angles: np.ndarray

    @overload
    def __init__(self, angles: np.ndarray) -> None:
        """
        Args:

            angles (numpy.ndarray): Spherical angles in radians.

        Raises:

            ValueError: On invalid argument types.
        """
        ...  # pragma: no cover

    @overload
    def __init__(self, azimuth: float, zenith: float) -> None:
        """
        Args:

            azimuth (float): Azimuth angle in radians.
            zenith (float): Zenith angle in radians.
        """
        ...  # pragma: no cover

    def __init__(  # type: ignore
        self, angles: float | np.ndarray, zenith: float | None = None
    ) -> None:
        if isinstance(angles, (float, int, np.float_, np.int_)):
            self.__angles = np.array([angles, zenith], dtype=np.float_)

        elif isinstance(angles, np.ndarray):
            self.__angles = angles

        else:
            raise ValueError("Invalid argument type")

        # Initialize base class
        BeamFocus.__init__(self)

    @property
    def angles(self) -> np.ndarray:
        """Spherical azimuth and zenith angles in radians."""

        return self.__angles

    def copy(self) -> SphericalFocus:
        return SphericalFocus(self.__angles.copy())

    @property
    def spherical_angles(self) -> np.ndarray:
        return self.__angles


class CoordinateFocus(BeamFocus):
    """Focus the beamformer towards a certain Cartesian coordinate."""

    yaml_tag = "CoordinateFocus"
    __direction: Direction
    __reference: Literal["global", "local"]

    def __init__(
        self, coordinates: np.ndarray | Direction, reference: Literal["global", "local"] = "local"
    ) -> None:
        """
        Args:

            coordinates (numpy.ndarray): Cartesian coordinates in m.
            reference (str, optional): Reference frame of the coordinates.
        """

        # Initialize base class
        BeamFocus.__init__(self)

        # Initialize class members
        self.__direction = (
            coordinates
            if isinstance(coordinates, Direction)
            else Direction.From_Cartesian(coordinates, True)
        )
        self.__reference = reference

    @property
    def coordinates(self) -> np.ndarray:
        """Cartesian coordinates in m."""

        return self.__direction.view(np.ndarray)

    @property
    def reference(self) -> Literal["global", "local"]:
        """Reference frame of the coordinates."""

        return self.__reference

    def copy(self) -> CoordinateFocus:
        return CoordinateFocus(self.__direction.copy(), self.__reference)

    @property
    def spherical_angles(self) -> np.ndarray:
        if self.reference == "local":
            return self.__direction.to_spherical()

        elif self.reference == "global":
            if self.beamformer is None:
                raise RuntimeError(
                    "Global coordinate focus requires the beamformer to be specified"
                )

            if self.beamformer.operator is None:
                raise RuntimeError(
                    "Global coordinate focus requires the beaformer to be assigned to an operator"
                )

            base = self.beamformer.operator.device
            if base is None:
                raise RuntimeError(
                    "Global coordinate focus requires the beamformer's operator to be assigned to a device"
                )

            transformation = base.antennas.backwards_transformation
            local_direction = transformation.transform_direction(self.__direction)
            return local_direction.to_spherical()


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

    def _assumed_array(self, array: AntennaArray | None) -> AntennaArray:
        """Infer the antenna array used for beamforming.

        Args:

            array (AntennaArray, optional):
                The assumed antenna array.
                If `None`, the operator's antenna array is used.

        Returns: The array.

        Raises:

            FloatingError: If the operator or operator device are not yet specified.
        """

        # Return the provided array
        if array is not None:
            return array

        if self.operator is None:
            raise FloatingError("Unable to steer a signal over a floating beamformer")

        if self.operator.device is None:
            raise FloatingError("Unable to steer a signal over a floating operator")

        return self.operator.device.antennas


class TransmitBeamformer(BeamformerBase[Transmitter], TransmitStreamEncoder, ABC):
    """Base class for beam steering precodings during signal transmissions."""

    __focus_points: Sequence[BeamFocus]

    def __init__(self, operator: Transmitter | None = None) -> None:
        """Args:

        operator (Transmitter, optional):
            The operator this beamformer is attached to.
            By default, the beamformer is considered floating.
        """

        self.__focus_points = [
            SphericalFocus(0.0, 0.0) for _ in range(self.num_transmit_focus_points)
        ]

        BeamformerBase.__init__(self, operator=operator)

    @abstractproperty
    def num_transmit_input_streams(self) -> int:
        """Number of input streams required by this beamformer.

        Returns:

            Number of input streams.
        """
        ...  # pragma: no cover

    @abstractproperty
    def num_transmit_output_streams(self) -> int:
        """Number of output streams generated by this beamformer.

        Returns:

            Number of output streams.
        """
        ...  # pragma: no cover

    @abstractproperty
    def num_transmit_focus_points(self) -> int:
        """Number of required transmit focus points.

        If this is :math:`1`,
        the beamformer is considered to be a single focus point beamformer and
        :attr:`.transmit_focus` will return a single focus point.
        Otherwise, the beamformer is considered a multi focus point beamformer and
        :attr:`.transmit_focus` will return a :py:obj:`Sequence` of focus points.

        Returns: Number of focus points.
        """
        ...  # pragma: no cover

    def encode_streams(self, streams: Signal) -> Signal:
        if streams.num_streams != self.num_transmit_input_streams:
            raise ValueError(
                "Stream encoding configuration invalid, number of provided streams don't match the beamformer requirements"
            )

        return self.transmit(streams)

    @TransmitStreamEncoder.precoding.setter  # type: ignore
    def precoding(self, precoding: Precoding) -> None:
        self.operator = precoding.modem  # type: ignore
        TransmitStreamEncoder.precoding.fset(self, precoding)  # type: ignore

    @abstractmethod
    def _encode(
        self,
        samples: np.ndarray,
        carrier_frequency: float,
        focus_angles: np.ndarray,
        array: AntennaArray,
    ) -> np.ndarray:
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

            array (AntennaArray):
                The assumed antenna array.
        """
        ...  # pragma: no cover

    @property
    def transmit_focus(self) -> BeamFocus | Sequence[BeamFocus]:
        """Focus points of the beamformer during transmission.

        Depending on :attr:`.num_transmit_focus_points`
        this is either a single focus point or a :py:obj:`Sequence` of points.

        Raises:

            ValueError: If the provided number of focus points does not match the number of required focus points.
        """

        return self.__focus_points

    @transmit_focus.setter
    def transmit_focus(self, value: BeamFocus | Sequence[BeamFocus]) -> None:
        # Force value to be a sequence internally
        _value = [value] if isinstance(value, BeamFocus) else [v for v in value]

        # Make sure the expected number of focus points is provided
        if len(_value) != self.num_transmit_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_value)} points, but the beamformer requires {self.num_transmit_focus_points} focus points"
            )

        # Set the focus points' beamformer reference
        for focus in _value:
            focus.beamformer = self

        # Save results
        self.__focus_points = _value

    def transmit(
        self,
        signal: Signal,
        focus: BeamFocus | Sequence[BeamFocus] | None = None,
        array: AntennaArray | None = None,
    ) -> Signal:
        """Focus a signal model towards a certain target.

        Args:

            signal (Signal):
                The signal to be steered.

            focus (BeamFocus | Sequence[BeamFocus], optional):
                Focus points of the steered signal power.
                If `None`, the beamformer's default :attr:`.transmit_focus` is used.

            array (AntennaArray, optional):
                Antenna array assumed used for steering.
                If `None`, the operator's antenna array is used.

        Returns:

            Samples of the focused signal.

        Raises:

            RuntimeError: If the operator or operator device are not yet specified.
            RuntimeError: If the number of signal streams does not match the number of required input streams.
            ValueError: If the number of focus points does not match the number of required focus points.
        """

        # Infer the required focus points
        _focus: Sequence[BeamFocus]
        if focus is None:
            _focus = (
                [self.transmit_focus]
                if isinstance(self.transmit_focus, BeamFocus)
                else self.transmit_focus
            )
        elif isinstance(focus, BeamFocus):
            _focus = [focus]
        else:
            _focus = focus

        # Infer the assumed antenna array
        _array = self._assumed_array(array)

        if signal.num_streams != self.num_transmit_input_streams:
            raise RuntimeError(
                f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_transmit_input_streams} streams"
            )

        if len(_focus) != self.num_transmit_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_focus)} points, but the beamformer requires {self.num_transmit_focus_points} focus points"
            )

        carrier_frequency = signal.carrier_frequency
        samples = signal[:, :]
        focus_angles = np.array([focus.spherical_angles for focus in _focus], dtype=np.float_)

        steered_samples = self._encode(samples, carrier_frequency, focus_angles, _array)
        return signal.from_ndarray(steered_samples)


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

    __focus_points: Sequence[BeamFocus]

    def __init__(self, operator: Receiver | None = None) -> None:
        """Args:

        operator (Receiver, optional):
            The operator this beamformer is attached to.
            By default, the beamformer is considered floating.
        """

        self.receive_focus = [
            SphericalFocus(0.0, 0.0) for _ in range(self.num_receive_focus_points)
        ]
        self.probe_focus_points = np.zeros((1, self.num_receive_focus_points, 2), dtype=float)

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
        ...  # pragma: no cover

    @abstractproperty
    def num_receive_output_streams(self) -> int:
        """Number of output streams generated by this beamformer.

        Dimension :math:`M` of the output sample matrix
        :math:`\\mathbf{Y} \\in \\mathbb{C}^{M \\times T}`.

        Returns:

            Number of output streams :math:`M`.
        """
        ...  # pragma: no cover

    @abstractproperty
    def num_receive_focus_points(self) -> int:
        """Number of required receive focus points.

        If this is :math:`1`,
        the beamformer is considered to be a single focus point beamformer and
        :attr:`.receive_focus` will return a single focus point.
        Otherwise, the beamformer is considered a multi focus point beamformer and
        :attr:`.receive_focus` will return a :py:obj:`Sequence` of focus points.

        Returns: Number of focus points.
        """
        ...  # pragma: no cover

    def decode_streams(self, streams: Signal) -> Signal:
        if streams.num_streams != self.num_receive_input_streams:
            raise ValueError(
                "Stream decoding configuration invalid, number of provided streams don't match the beamformer requirements"
            )

        return self.receive(streams)

    @ReceiveStreamDecoder.precoding.setter  # type: ignore
    def precoding(self, precoding: Precoding) -> None:
        self.operator = precoding.modem  # type: ignore
        ReceiveStreamDecoder.precoding.fset(self, precoding)  # type: ignore

    @abstractmethod
    def _decode(
        self, samples: np.ndarray, carrier_frequency: float, angles: np.ndarray, array: AntennaArray
    ) -> np.ndarray:
        """Decode signal streams for receive beamforming.

        This method is called as a subroutine during :meth:`receive` and :meth:`probe`.

        Args:

            samples (np.ndarray):
                Signal samples, first dimension being the number of signal streams :math:`N`, second the number of samples :math:`T`.

            carrier_frequency (float):
                The assumed carrier central frequency of the samples :math:`f_\\mathrm{c}`.

            angles (numpy.ndarray):
                Spherical coordinate system angles of arrival in radians.
                A three-dimensional numpy array with the first dimension representing the number of angles,
                the second dimension of magnitude number of focus points :math:`F`,
                and the third dimension containing the azimuth and zenith angle in radians, respectively.

            array (AntennaArray):
                The assumed antenna array.

        Returns:

            Stream samples of the focused signal towards all focus points.
            A three-dimensional numpy array with the first dimension representing the number of focus points,
            the second dimension the number of returned streams and the third dimension the amount of samples.
        """
        ...  # pragma: no cover

    @property
    def receive_focus(self) -> BeamFocus | Sequence[BeamFocus]:
        """Focus points of the beamformer during reception.

        Depending on :attr:`.num_receive_focus_points`
        this is either a single focus point or a :py:obj:`Sequence` of points.

        Raises:

            ValueError: If the provided number of focus points does not match the number of required focus points.
        """

        return self.__focus_points

    @receive_focus.setter
    def receive_focus(self, value: BeamFocus | Sequence[BeamFocus]) -> None:
        # Force value to be a sequence internally
        _value = [value] if isinstance(value, BeamFocus) else [v for v in value]

        # Make sure the expected number of focus points is provided
        if len(_value) != self.num_receive_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_value)} points, but the beamformer requires {self.num_receive_focus_points} focus points"
            )

        # Set the focus points' beamformer reference
        for focus in _value:
            focus.beamformer = self

        # Save results
        self.__focus_points = _value

    def receive(
        self,
        signal: Signal,
        focus: BeamFocus | Sequence[BeamFocus] | None = None,
        array: AntennaArray | None = None,
    ) -> Signal:
        """Focus a signal model towards a certain target.

        Args:

            signal (Signal):
                The signal to be steered.

            focus (BeamFocus | Sequence[BeamFocus], optional):
                Focus of the steered signal power.
                If not provided, the beamformer's default :attr:`.receive_focus` is used.

            array (AntennaArray, optional):
                Antenna array assumed used for steering.
                If not specified, the operator's antenna array is used.

        Returns:

            Signal focused towards the requested focus points.

        Raises:

            FloatingError: If the operator or operator device are not yet specified.
            RuntimeError: If the number of signal streams does not match the number of required input streams.
            ValueError: If the number of focus points does not match the number of required focus points.
        """

        # Infer the required focus points
        _focus: Sequence[BeamFocus]
        if focus is None:
            _focus = (
                [self.receive_focus]
                if isinstance(self.receive_focus, BeamFocus)
                else self.receive_focus
            )
        elif isinstance(focus, BeamFocus):
            _focus = [focus]
        else:
            _focus = focus

        # Infer the assumed antenna array
        _array = self._assumed_array(array)

        if array is None and signal.num_streams != self.num_receive_input_streams:
            raise RuntimeError(
                f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_receive_input_streams} streams"
            )

        if len(_focus) != self.num_receive_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_focus)} points, but the beamformer requires {self.num_receive_focus_points} focus points"
            )

        carrier_frequency = signal.carrier_frequency
        samples = signal[:, :]
        focus_angles = np.array([[focus.spherical_angles for focus in _focus]], dtype=np.float_)

        beamformed_samples = self._decode(samples, carrier_frequency, focus_angles, _array)
        return signal.from_ndarray(beamformed_samples[0, ::])

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

        if value.shape[1] != self.num_receive_focus_points:
            raise ValueError(
                f"Focus requires {self.num_receive_focus_points} points, but {value.shape[1]} were provided"
            )

        self.__probe_focus_points = value

    def probe(self, signal: Signal, focus_points: np.ndarray | None = None) -> np.ndarray:
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
            raise RuntimeError(
                f"The provided signal contains {signal.num_streams}, but the beamformer requires {self.num_receive_input_streams} streams"
            )

        carrier_frequency = signal.carrier_frequency
        samples = signal[:, :]

        return self._decode(samples, carrier_frequency, focus_points, self.operator.device.antennas)
