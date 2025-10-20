# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing import Generic, Sequence, Tuple, TypeVar
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light

from hermespy.beamforming import ReceiveBeamformer
from hermespy.core import (
    DenseSignal,
    DeserializationProcess,
    FloatingError,
    Signal,
    Serializable,
    Transmission,
    TransmitState,
    Transmitter,
    Receiver,
    ReceiveState,
    Reception,
    SerializationProcess,
)
from .cube import RadarCube
from .detection import RadarDetector, RadarPointCloud

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarWaveform(Serializable):
    """Base class for :class:`.Radar` waveform descriptions.

    When assigned to a :class:`.Radar`'s :meth:`waveform<.Radar.waveform>` property,
    the waveform's :meth:`.ping` and :meth:`.estimate` methods are called as subroutines
    of the :class:`.Radar`'s :meth:`.Radar._transmit` and :meth:`.Radar._receive` routines, respectively.


    .. mermaid::

        classDiagram

            class Radar {
                +waveform : RadarWaveform
                +transmit() : RadarTransmission
                +receive() : RadarReception
            }

            class RadarWaveform {
                <<Abstract>>
                +sampling_rate: float*
                +frame_duration: float*
                +max_range: float*
                +range_resolution: float*
                +range_bins: ndarray
                +max_relative_doppler: float*
                +relative_doppler_resolution: float*
                +energy: float*
                +power: float*
                +ping() : Signal*
                +estimate(Signal) : ndarray*

            }

            class FMCW {
                +sampling_rate: float
                +frame_duration: float
                +max_range: float
                +range_resolution: float
                +range_bins: ndarray
                +max_relative_doppler: float
                +relative_doppler_resolution: float
                +energy: float
                +power: float
                +ping() : Signal
                +estimate(Signal) : ndarray

            }

            Radar *-- RadarWaveform
            FMCW ..|> RadarWaveform
            Radar --> RadarWaveform: ping()
            Radar --> RadarWaveform: estimate()

            link RadarWaveform "#hermespy.radar.radar.RadarWaveform"
            link Radar "radar.Radar.html"
            link FMCW "fmcw.html"

    The currently available radar waveforms are:

    .. toctree::

        fmcw
    """

    @abstractmethod
    def ping(self, state: TransmitState) -> Signal:
        """Generate a single radar frame.

        Args:
            state: State of the device the radar is assigned to.

        Returns:
            Single-stream signal model of a single radar frame.
        """
        ...  # pragma: no cover

    @abstractmethod
    def estimate(self, signal: Signal, state: ReceiveState) -> np.ndarray:
        """Generate a range-doppler map from a single-stream radar frame.

        Args:
            signal: Single-stream signal model of a single propagated radar frame.
            state: State of the device the radar is assigned to.

        Returns:
            Numpy matrix (2D array) of the range-doppler map, where the first dimension indicates
            discrete doppler frequency bins and the second dimension indicates discrete range bins.
        """
        ...  # pragma: no cover

    @abstractmethod
    def frame_duration(self, bandwidth: float) -> float:
        """Duration of a single radar frame in seconds.

        Args:
            bandwidth: Bandwidth of the radar waveform in Hz.

        Denoted by :math:`T_{\\mathrm{F}}` of unit :math:`\\left[ T_{\\mathrm{F}} \\right] = \\mathrm{s}` in literature.
        """
        ...  # pragma: no cover

    @abstractmethod
    def max_range(self, bandwidth: float) -> float:
        """The waveform's maximum detectable range in meters.

        Args:
            bandwidth: Bandwidth of the radar waveform in Hz.

        Denoted by :math:`R_{\\mathrm{Max}}` of unit :math:`\\left[ R_{\\mathrm{Max}} \\right] = \\mathrm{m}` in literature.
        """
        ...  # pragma: no cover

    @abstractmethod
    def range_resolution(self, bandwidth: float) -> float:
        """Resolution of the radial range sensing in meters.

        Args:
            bandwidth: Bandwidth of the radar waveform in Hz.

        Denoted by :math:`\\Delta R` of unit :math:`\\left[ \\Delta R \\right] = \\mathrm{m}` in literature.
        """
        ...  # pragma: no cover

    @abstractmethod
    def range_bins(self, bandwidth: float) -> np.ndarray:
        """Discrete sample bins of the radial range sensing.

        Args:
            bandwidth: Bandwidth of the radar waveform in Hz.

        Returns:
            A numpy vector (1D array) of discrete range bins in meters.
        """

        resolution = self.range_resolution(bandwidth)
        return np.arange(int(self.max_range(bandwidth) / resolution)) * resolution

    @property
    @abstractmethod
    def max_relative_doppler(self) -> float:
        """Maximum relative detectable radial doppler frequency shift in Hz.

        .. math::

           \\Delta f_\\mathrm{Max} = \\frac{v_\\mathrm{Max}}{\\lambda}
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def relative_doppler_resolution(self) -> float:
        """Relative resolution of the radial doppler frequency shift sensing in Hz.

        .. math::

           \\Delta f_\\mathrm{Res} = \\frac{v_\\mathrm{Res}}{\\lambda}
        """
        ...  # pragma: no cover

    @property
    def relative_doppler_bins(self) -> np.ndarray:
        """Realtive discrete sample bins of the radial doppler frequency shift sensing.

        A numpy vector (1D array) of discrete doppler frequency bins in Hz.
        """
        ...  # pragma: no cover

    @abstractmethod
    def energy(self, bandwidth: float, oversampling_factor: int) -> float:
        """Energy of the radar waveform.

        Args:
            bandwidth: Bandwidth of the radar waveform in Hz.
            oversampling_factor: Oversampling factor of the radar waveform.

        Radar energy in :math:`\\mathrm{Wh}`.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def power(self) -> float:
        """Power of the radar waveform.

        Radar power in :math:`\\mathrm{W}`.
        """
        ...  # pragma: no cover

    @abstractmethod
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        """Number of samples in a single radar frame.

        Args:
            bandwidth: Bandwidth of the radar waveform in Hz.
            oversampling_factor: Oversampling factor of the radar waveform.

        Returns:
            Number of samples in a single radar frame.
        """
        ...  # pragma: no cover


class RadarTransmission(Transmission):
    """Information generated by transmitting over a radar.
    Generated by calling a :class:`.Radar`'s :meth:`transmit()<.Radar._transmit>` method.
    """

    def __init__(self, signal: Signal) -> None:
        """
        Args:

            signal: Transmitted radar waveform.
        """

        Transmission.__init__(self, signal)


class RadarReception(Reception):
    """Information generated by receiving over a radar.

    Generated by calling a :class:`.Radar`'s :meth:`_receive<.Radar._receive>` method.
    """

    __cube: RadarCube  # Processed raw radar data
    __cloud: RadarPointCloud | None  # Detected radar point cloud

    def __init__(
        self, signal: Signal, cube: RadarCube, cloud: RadarPointCloud | None = None
    ) -> None:
        """
        Args:

            signal: Received radar waveform.
            cube: Processed raw radar data.
            cloud: Radar point cloud. :py:obj:`None` if a point cloud was not generated.
        """

        # Initialize base class
        Reception.__init__(self, signal)

        # Initialize class attributes
        self.__cube = cube
        self.__cloud = cloud

    @property
    def cube(self) -> RadarCube:
        """Cube of processed raw radar data."""

        return self.__cube

    @property
    def cloud(self) -> RadarPointCloud | None:
        """Detected radar point cloud.

        :py:obj:`None` if a point cloud was not generated.
        """

        return self.__cloud

    @override
    def serialize(self, process: SerializationProcess) -> None:
        Reception.serialize(self, process)
        process.serialize_object(self.__cube, "cube")
        if self.cloud is not None:
            process.serialize_object(self.cloud, "cloud")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> RadarReception:
        signal = process.deserialize_object("signal", Signal)
        cube = process.deserialize_object("cube", RadarCube)
        cloud = process.deserialize_object("cloud", RadarPointCloud, None)
        return RadarReception(signal, cube, cloud)


RTT = TypeVar("RTT", bound=RadarTransmission)
"""Type of radar transmission."""

RRT = TypeVar("RRT", bound=RadarReception)
"""Type of radar reception."""


class RadarBase(Generic[RTT, RRT], Transmitter[RTT], Receiver[RRT]):
    """Base class class for radar sensing signal processing pipelines."""

    _DEFAULT_MIN_RANGE: float = 0.0

    __receive_beamformer: ReceiveBeamformer | None
    __detector: RadarDetector | None
    __min_range: float

    def __init__(
        self,
        receive_beamformer: ReceiveBeamformer | None = None,
        detector: RadarDetector | None = None,
        min_range: float = _DEFAULT_MIN_RANGE,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        carrier_frequency: float | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            receive_beamformer:
                Beamforming applied during signal reception.
                If not specified, no beamforming will be applied during reception.

            detector:
                Detector routine configured to generate point clouds from radar cubes.
                If not specified, no point cloud will be generated during reception.

            min_range:
                Minimal range considered for the generated radar cubes.
                Zero by default, but can be adjusted to ignore, for example, self-interference.

            selected_transmit_ports:
                Indices of antenna ports selected for transmission from the operated :class:`Device's<hermespy.core.device.Device>` antenna array.
                If not specified, all available ports will be considered.

            selected_receive_ports:
                Indices of antenna ports selected for reception from the operated :class:`Device's<hermespy.core.device.Device>` antenna array.
                If not specified, all available antenna ports will be considered.

            carrier_frequency:
                Central frequency of the mixed signal in radio-frequency transmission band.
                If not specified, the operated device's default carrier frequency will be assumed during signal processing.

            seed:
                Random seed used to initialize the pseudo-random number generator.
        """

        # Initialize base classes
        Transmitter.__init__(self, seed, selected_transmit_ports, carrier_frequency)
        Receiver.__init__(self, seed, selected_receive_ports, carrier_frequency)

        # Initialize class attributes
        self.receive_beamformer = receive_beamformer
        self.detector = detector
        self.min_range = min_range

    @property
    def receive_beamformer(self) -> ReceiveBeamformer | None:
        """Beamforming applied during signal reception.

        The :class:`TransmitBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>`'s :meth:`probe<hermespy.beamforming.beamformer.ReceiveBeamformer.probe>`
        method is called as a subroutine of :meth:`Receiver.receive()<hermespy.core.device.Receiver.receive>`.

        Configuration is required if the assigned :class:`Device<hermespy.core.device.Device>` features multiple :meth:`antennas<hermespy.core.device.Device.antennas>`.
        """

        return self.__receive_beamformer

    @receive_beamformer.setter
    def receive_beamformer(self, value: ReceiveBeamformer | None) -> None:
        self.__receive_beamformer = value

    @property
    def detector(self) -> RadarDetector | None:
        """Detector routine configured to generate point clouds from radar cubes.

        If configured, during :meth:`_receive<Radar._receive>`,
        the detector's :meth:`detect<hermespy.radar.detection.RadarDetector.detect>` method is called to generate a :class:`RadarPointCloud<hermespy.radar.detection.RadarPointCloud>`.
        If not configured, i.e. :obj:`None`, the generated :class:`.RadarReception`'s :attr:`cloud<.RadarReception.cloud>` property will be :py:obj:`None`.
        """

        return self.__detector

    @detector.setter
    def detector(self, value: RadarDetector | None) -> None:
        self.__detector = value

    @property
    def min_range(self) -> float:
        """Minimum detection range.

        Can be adjusted to ignore, for example, self-interference.

        Raises:

            ValueError: For negative values.
        """

        return self.__min_range

    @min_range.setter
    def min_range(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Minimum range must be non-negative")

        self.__min_range = value

    def _receive_beamform(
        self, signal: Signal, device: ReceiveState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply digital beamforming to the received signal.

        Subroutine of :meth:`_receive<Radar._receive>`.

        Args:

            signal: Received radar waveform.
            device: State of the device the radar is assigned to.

        Returns:
            Tuple of the angles of interest and the beamformed samples.

        Raises:

            RuntimeError: If the device has more than one antenna, but no beamforming strategy is configured.
            RuntimeError: If the beamforming configuration does not result in a single output stream.
        """

        if signal.num_streams > 1:
            if self.receive_beamformer is None:
                raise RuntimeError(
                    "Receiving over a device with more than one RF port requires a beamforming configuration"
                )

            num_receveive_output_streams = self.receive_beamformer.num_receive_output_streams(
                signal.num_streams
            )
            if num_receveive_output_streams < 0:
                raise RuntimeError(
                    f"Configured radar receive beamformer does not support {signal.num_streams} input streams"
                )
            elif num_receveive_output_streams != 1:
                raise RuntimeError(
                    "Only receive beamformers generating a single output stream are supported by radar operators"
                )

            beamformed_samples = self.receive_beamformer.probe(signal.view(np.ndarray), device)[
                :, 0, :
            ]

        else:
            beamformed_samples = signal.view(np.ndarray)

        # Build the radar cube by generating a beam-forming line over all angles of interest
        angles_of_interest = (
            np.array([[0.0, 0.0]], dtype=float)
            if self.receive_beamformer is None
            else self.receive_beamformer.probe_focus_points[:, 0, :]
        )

        return angles_of_interest, beamformed_samples

    @override
    def serialize(self, process: SerializationProcess) -> None:
        Receiver.serialize(self, process)
        Transmitter.serialize(self, process)
        if self.receive_beamformer is not None:
            process.serialize_object(self.receive_beamformer, "receive_beamformer")
        if self.detector is not None:
            process.serialize_object(self.detector, "detector")
        process.serialize_floating(self.min_range, "min_range")

    @classmethod
    @override
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        params = Receiver._DeserializeParameters(process)
        params.update(Transmitter._DeserializeParameters(process))
        params.update(
            {
                "receive_beamformer": process.deserialize_object(
                    "receive_beamformer", ReceiveBeamformer, None
                ),
                "detector": process.deserialize_object("detector", RadarDetector, None),
                "min_range": process.deserialize_floating("min_range", cls._DEFAULT_MIN_RANGE),
            }
        )
        return params


class Radar(RadarBase[RadarTransmission, RadarReception], Serializable):
    """Signal processing pipeline of a monostatic radar sensing its environment.

    The radar can be configured by assigning four composite objects to respective property attributes:

    .. list-table::
       :header-rows: 1

       * - Property
         - Type

       * - :meth:`waveform<.waveform>`
         - :class:`RadarWaveform`

       * - :meth:`receive_beamformer<.receive_beamformer>`
         - :class:`ReceiveBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>`

       * - :meth:`detector<.detector>`
         - :class:`RadarDetector<hermespy.radar.detection.RadarDetector>`

    Of those, only a :class:`RadarWaveform` is mandatory.
    Beamformers, i.e. a :class:`TransmitBeamformer<hermespy.beamforming.beamformer.TransmitBeamformer>` and a :class:`ReceiveBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>`,
    are only required when the radar is assigned to a :class:`Device<hermespy.core.device.Device>` configured to multiple :meth:`antennas<hermespy.core.device.Device.antennas>`.
    A :class:`RadarDetector<hermespy.radar.detection.RadarDetector>` is optional,
    if not configured the radar's generated :class:`RadarReception` will not contain a :class:`RadarPointCloud<hermespy.radar.detection.RadarPointCloud>`.

    When assigned to a :class:`Device<hermespy.core.device.Device>`,
    device transmission will trigger the radar to generate a :class:`RadarTransmission` by executing the following sequence of calls:

    .. mermaid::

        sequenceDiagram

        participant Device
        participant Radar
        participant RadarWaveform
        participant TransmitBeamformer

        Device ->> Radar: _transmit()
        Radar ->> RadarWaveform: ping()
        RadarWaveform -->> Radar: Signal
        Radar ->>  TransmitBeamformer: transmit(Signal)
        TransmitBeamformer -->> Radar: Signal
        Radar -->> Device: RadarTransmission

    Initially, the :meth:`ping<RadarWaveform.ping>` method of the :class:`RadarWaveform` is called to generate the model
    of a single-antenna radar frame.
    For :class:`Devices<hermespy.core.device.Device>` configured to multiple :meth:`antennas<hermespy.core.device.Device.antennas>`,
    the configured :class:`TransmitBeamformer<hermespy.beamforming.beamformer.TransmitBeamformer>` is called to encode the signal for each antenna.
    The resulting multi-antenna frame, contained within the return :class:`RadarTransmission`, is cached at the assigned :class:`Device<hermespy.core.device.Device>`.

    When assigned to a :class:`Device<hermespy.core.device.Device>`,
    device reception will trigger the radar to generate a :class:`RadarReception` by executing the following sequence of calls:

    .. mermaid::

        sequenceDiagram

            participant Device
            participant Radar
            participant ReceiveBeamformer
            participant RadarWaveform
            participant RadarDetector

            Device ->> Radar: _receive(Signal)
            Radar ->> ReceiveBeamformer: probe(Signal)
            ReceiveBeamformer -->> Radar: line_signals
            loop
                Radar ->> RadarWaveform: estimate(line_signal)
                RadarWaveform -->> Radar: line_estimate
            end
            Radar ->> RadarDetector: detect(line_estimates)
            RadarDetector -->> Radar: RadarPointCloud
            Radar -->> Device: RadarReception

    Initially, the :meth:`probe<hermespy.beamforming.beamformer.ReceiveBeamformer.probe>` method of the :class:`ReceiveBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>` is called to generate a sequence of
    line signals from each direction of interest.
    We refer to them as *line signals* as they are the result of an antenna arrays beamforing towards a single direction of interest,
    so the signal can be though of as having propagated along a single line pointing towards the direction of interest.
    This step is only executed for :class:`Devices<hermespy.core.device.Device>` configured to multiple :meth:`antennas<hermespy.core.device.Device.antennas>`.
    The sequence of line signals are then indiviually processed by the :meth:`estimate<RadarWaveform.estimate>` method of the :class:`RadarWaveform`,
    resulting in a line estimate representing a range-doppler map for each direction of interest.
    This sequence line estimates is combined to a single :class:`RadarCube<hermespy.radar.cube.RadarCube>`.
    If a :class:`RadarDetector<hermespy.radar.detection.RadarDetector>` is configured, the :meth:`detect<hermespy.radar.detection.RadarDetector.detect>` method is called to generate a :class:`RadarPointCloud<hermespy.radar.detection.RadarPointCloud>`
    from the :class:`RadarCube<hermespy.radar.cube.RadarCube>`.
    The resulting information is cached as a :class:`RadarReception<hermespy.radar.radar.RadarReception>` at the assigned :class:`Device<hermespy.core.device.Device>`.
    """

    __waveform: RadarWaveform | None

    def __init__(
        self,
        waveform: RadarWaveform | None = None,
        receive_beamformer: ReceiveBeamformer | None = None,
        detector: RadarDetector | None = None,
        min_range: float = RadarBase._DEFAULT_MIN_RANGE,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        carrier_frequency: float | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            waveform:
                Description of the waveform to be transmitted and received by this radar.
                :py:obj:`None` if no waveform is configured.

            receive_beamformer:
                Beamforming applied during signal reception.
                If not specified, no beamforming will be applied during reception.

            detector:
                Detector routine configured to generate point clouds from radar cubes.
                If not specified, no point cloud will be generated during reception.

            min_range:
                Minimal range considered for the generated radar cubes.
                Zero by default, but can be adjusted to ignore, for example, self-interference.

            selected_transmit_ports:
                Indices of antenna ports selected for transmission from the operated :class:`Device's<hermespy.core.device.Device>` antenna array.
                If not specified, all available ports will be considered.

            selected_receive_ports:
                Indices of antenna ports selected for reception from the operated :class:`Device's<hermespy.core.device.Device>` antenna array.
                If not specified, all available antenna ports will be considered.

            carrier_frequency:
                Central frequency of the mixed signal in radio-frequency transmission band.
                If not specified, the operated device's default carrier frequency will be assumed during signal processing.

            seed:
                Random seed used to initialize the pseudo-random number generator.
        """

        # Initialize base classes
        RadarBase.__init__(
            self,
            receive_beamformer,
            detector,
            min_range,
            selected_transmit_ports,
            selected_receive_ports,
            carrier_frequency,
            seed,
        )

        # Initialize class attributes
        self.__waveform = None
        self.waveform = waveform

    @property
    def power(self) -> float:
        if self.waveform is None:
            return 0.0

        return self.waveform.power

    @override
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        if self.waveform is None:
            return 0

        return self.waveform.samples_per_frame(bandwidth, oversampling_factor)

    @property
    def waveform(self) -> RadarWaveform | None:
        """Description of the waveform to be transmitted and received by this radar.

        :py:obj:`None` if no waveform is configured.

        During :meth:`_transmit<hermespy.radar.radar.Radar._transmit>`,
        the :class:`.RadarWaveform`'s :meth:`ping()<.RadarWaveform.ping>` method is called
        to generate a signal to be transmitted by the radar.

        During :meth:`_receive<hermespy.radar.radar.Radar._receive>`, the :class:`.RadarWaveform`'s :meth:`estimate()<.RadarWaveform.estimate>` method is called
        multiple times to generate range-doppler line estimates for each direction of interest.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: RadarWaveform | None) -> None:
        self.__waveform = value

    def max_range(self, bandwidth: float) -> float:
        """The radar's maximum detectable range in meters.

        Args:
            bandwidth: Bandwidth of the radar waveform in Hz.

        Denoted by :math:`R_{\\mathrm{Max}}` of unit :math:`\\left[ R_{\\mathrm{Max}} \\right] = \\mathrm{m}` in literature.
        Convenience property that resolves to the configured :class:`.RadarWaveform`'s :meth:`max_range<.RadarWaveform.max_range>` property.
        Returns :math:`R_{\\mathrm{Max}} = 0` if no waveform is configured.
        """

        return 0.0 if self.waveform is None else self.waveform.max_range(bandwidth)

    def velocity_resolution(self, carrier_frequency: float) -> float:
        """The radar's velocity resolution in meters per second.

        Denoted by :math:`\\Delta v` of unit :math:`\\left[ \\Delta v \\right] = \\frac{\\mathrm{m}}{\\mathrm{s}}` in literature.
        Computed as

        .. math::

            \\Delta v = \\frac{c_0}{f_{\\mathrm{c}}} \\Delta f_{\\mathrm{Res}}

        querying the configured :class:`.RadarWaveform`'s :meth:`relative_doppler_resolution<.RadarWaveform.relative_doppler_resolution>` property :math:`\\Delta f_{\\mathrm{Res}}`.
        """

        if self.waveform is None:
            raise FloatingError("Cannot compute velocity resolution without a waveform")

        if carrier_frequency == 0.0:
            raise RuntimeError("Cannot compute velocity resolution in base-band carrier frequency")

        return 0.5 * self.waveform.relative_doppler_resolution * speed_of_light / carrier_frequency

    @override
    def _transmit(self, state: TransmitState, duration: float) -> RadarTransmission:
        if not self.__waveform:
            raise RuntimeError("Radar waveform not specified")

        # Generate the radar waveform
        signal = self.waveform.ping(state)

        # Radar only supports a single output stream
        if state.num_transmit_dsp_ports > 1:
            raise RuntimeError(
                "Radars only provide a single DSP output stream. Configure a transmit beamformer at the assigned device."
            )

        # Prepare transmission
        transmission = RadarTransmission(signal)

        return transmission

    @override
    def _receive(self, signal: Signal, state: ReceiveState) -> RadarReception:
        if not self.waveform:
            raise RuntimeError("Radar waveform not specified")

        # Resample signal properly
        # signal = signal.resample(self.__waveform.sampling_rate)

        # If the device has more than one antenna, a beamforming strategy is required
        angles_of_interest, beamformed_samples = self._receive_beamform(signal, state)
        range_bins = self.waveform.range_bins(state.bandwidth)
        doppler_bins = self.waveform.relative_doppler_bins

        # Filter the cube data to only contain the range bins outside the considered minimal range
        selected_range_idxs = np.where(range_bins >= self.min_range)[0]
        selected_range_bins = range_bins[selected_range_idxs]

        cube_data = np.empty(
            (len(angles_of_interest), len(doppler_bins), len(selected_range_bins)), dtype=float
        )

        for angle_idx, line in enumerate(beamformed_samples):
            # Process the single angular line by the waveform generator
            line_signal = DenseSignal.FromNDArray(line[None, :], state.sampling_rate)
            line_estimate = self.waveform.estimate(line_signal, state)

            cube_data[angle_idx, ::] = line_estimate[:, selected_range_idxs]

        # Create radar cube object
        cube = RadarCube(
            cube_data,
            angles_of_interest,
            doppler_bins,
            selected_range_bins,
            state.carrier_frequency,
        )

        # Infer the point cloud, if a detector has been configured
        cloud = None if self.detector is None else self.detector.detect(cube)

        reception = RadarReception(signal, cube, cloud)
        return reception

    @override
    def serialize(self, process: SerializationProcess) -> None:
        RadarBase.serialize(self, process)
        if self.waveform is not None:
            process.serialize_object(self.waveform, "waveform")

    @classmethod
    @override
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        params = RadarBase._DeserializeParameters(process)
        params["waveform"] = process.deserialize_object("waveform", RadarWaveform, None)
        return params
