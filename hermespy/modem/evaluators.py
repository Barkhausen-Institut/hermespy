# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

from hermespy.core import (
    ArtifactTemplate,
    DeserializationProcess,
    Serializable,
    SerializationProcess,
    Evaluator,
    EvaluationTemplate,
    GridDimension,
    Hook,
    PlotVisualization,
    ScalarEvaluationResult,
    StemVisualization,
    VAT,
)
from .modem import (
    CommunicationReception,
    CommunicationTransmission,
    TransmittingModem,
    ReceivingModem,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CommunicationEvaluator(Evaluator, Serializable):
    """Base class for evaluating communication processes between two modems."""

    _DEFAULT_PLOT_SURFACE: bool = True

    __transmitting_modem: TransmittingModem
    __receiving_modem: ReceivingModem
    __transmit_hook: Hook[CommunicationTransmission]
    __receive_hook: Hook[CommunicationReception]
    __transmission: CommunicationTransmission | None
    __reception: CommunicationReception | None
    __plot_surface: bool

    def __init__(
        self,
        transmitting_modem: TransmittingModem,
        receiving_modem: ReceivingModem,
        plot_surface: bool = _DEFAULT_PLOT_SURFACE,
    ) -> None:
        """
        Args:

            transmitting_modem (TransmittingModem):
                Communication modem transmitting information.

            receiving_modem (ReceivingModem):
                Communication modem receiving information.

            plot_surface (bool, optional):
                Plot the surface of the evaluation result in two-dimensional grids.
                Defaults to True.
        """

        # Initialize base class
        Evaluator.__init__(self)

        # Initialize class attributes
        self.__transmitting_modem = transmitting_modem
        self.__receiving_modem = receiving_modem
        self.__transmission = None
        self.__reception = None
        self.__plot_surface = plot_surface

        # Register callbacks for new transmissions and receptions
        self.__transmit_hook = transmitting_modem.add_transmit_callback(self.__transmit_callback)
        self.__receive_hook = receiving_modem.add_receive_callback(self.__receive_callback)

    @property
    def transmitting_modem(self) -> TransmittingModem:
        """Communication modem transmitting information."""

        return self.__transmitting_modem

    @property
    def receiving_modem(self) -> ReceivingModem:
        """Communication modem receiving information."""

        return self.__receiving_modem

    def __transmit_callback(self, transmission: CommunicationTransmission) -> None:
        """Callback function notifying the evaluator of a new transmission."""

        self.__transmission = transmission

    def __receive_callback(self, reception: CommunicationReception) -> None:
        """Callback function notifying the evaluator of a new reception."""

        self.__reception = reception

    def _fetch_dsp_results(self) -> tuple[CommunicationTransmission, CommunicationReception]:
        """Fetches the cached communication transmission and reception results.

        Raises:

            RuntimeError: If the transmission or reception results are not available.

        Returns: Tuple of transmission and reception.
        """

        if self.__transmission is None:
            raise RuntimeError(
                "Communication evaluator could not fetch transmission. Has the modem transmitted data?"
            )

        if self.__reception is None:
            raise RuntimeError(
                "Communication evaluator could not fetch reception. Has the modem received data?"
            )

        return self.__transmission, self.__reception

    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> ScalarEvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self, self.__plot_surface)

    def __del__(self) -> None:
        """Destructor of the communication evaluator.

        Ensures that the hooks are removed when the evaluator is deleted.
        """

        self.__transmit_hook.remove()
        self.__receive_hook.remove()

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__transmitting_modem, "transmitting_modem")
        process.serialize_object(self.__receiving_modem, "receiving_modem")
        process.serialize_integer(self.__plot_surface, "plot_surface")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> CommunicationEvaluator:
        return cls(
            process.deserialize_object("transmitting_modem", TransmittingModem),
            process.deserialize_object("receiving_modem", ReceivingModem),
            bool(process.deserialize_integer("plot_surface", cls._DEFAULT_PLOT_SURFACE)),
        )


class ErrorEvaluation(EvaluationTemplate[np.ndarray, StemVisualization], ABC):
    """Base class for error evaluations between two modems exchanging information."""

    @property
    @abstractmethod
    def _x_axis_label(self) -> str:
        """Label of the visualization's x-axis."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def _y_axis_label(self) -> str:
        """Label of the visualization's y-axis."""
        ...  # pragma: no cover

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> StemVisualization:
        # Configure axes
        axes[0, 0].set_ylim([-0.1, 1.1])
        axes[0, 0].set_xlabel(self._x_axis_label)
        axes[0, 0].set_ylabel(self._y_axis_label)
        axes[0, 0].set_yticks([])

        # Plot horizone lines for reference indicators
        axes[0, 0].axhline(0.0, linestyle="--")
        axes[0, 0].axhline(1.0, linestyle="--")
        axes[0, 0].text(-1.0, -0.06, "Correct", fontweight="bold")
        axes[0, 0].text(-1.0, 1.06, "Error", fontweight="bold")

        stem = axes[0, 0].stem(np.zeros(self.evaluation.size), basefmt=" ")
        return StemVisualization(figure, axes, stem)

    def _update_visualization(self, visualization: StemVisualization, **kwargs) -> None:
        # Update markers
        visualization.container.markerline.set_ydata(self.evaluation)

        # ToDo: Update segemts representing the stem lines
        # for line, bit in zip(visualization.container.stemlines, self.evaluation.flat):
        #   line.set_ydata(bit)


class BitErrorArtifact(ArtifactTemplate[np.float64]):
    """Artifact of a bit error evaluation between two modems exchanging information.

    Generated by :meth:`artifact()<BitErrorEvaluation.artifact>` of :class:`BitErrorEvaluation`.
    """

    ...  # pragma: no cover


class BitErrorEvaluation(ErrorEvaluation):
    """Bit error evaluation between two modems exchanging information.

    Generated by :meth:`evaluate()<BitErrorEvaluator.evaluate>` of :class:`BitErrorEvaluator`.
    """

    @property
    def title(self) -> str:
        return "Bit Error Evaluation"

    @property
    def _x_axis_label(self) -> str:
        return "Bit Index"

    @property
    def _y_axis_label(self) -> str:
        return "Bit Error Indicator"

    def artifact(self) -> BitErrorArtifact:
        ber = np.mean(self.evaluation)
        return BitErrorArtifact(ber)


class BitErrorEvaluator(CommunicationEvaluator, Serializable):
    """Evaluate bit errors between two modems exchanging information."""

    def __init__(
        self,
        transmitting_modem: TransmittingModem,
        receiving_modem: ReceivingModem,
        plot_surface: bool = True,
    ) -> None:
        """
        Args:

            transmitting_modem (TransmittingModem):
                Modem transmitting information.

            receiving_modem (ReceivingModem):
                Modem receiving information.

            plot_surface (bool, optional):
                Plot the surface of the evaluation result in two-dimensional grids.
                Defaults to True.
        """

        CommunicationEvaluator.__init__(self, transmitting_modem, receiving_modem, plot_surface)
        self.plot_scale = "log"  # Plot logarithmically by default

    def evaluate(self) -> BitErrorEvaluation:
        # Retrieve transmitted and received bits
        transmission, reception = self._fetch_dsp_results()
        transmitted_bits = transmission.bits
        received_bits = reception.bits

        # Pad bit sequences (if required)
        num_bits = max(len(received_bits), len(transmitted_bits))
        padded_transmission = np.append(
            transmitted_bits, np.zeros(num_bits - len(transmitted_bits))
        )
        padded_reception = np.append(received_bits, np.zeros(num_bits - len(received_bits)))

        # Compute bit errors as the positions where both sequences differ.
        # Note that this requires the sequences to be in 0/1 format!
        bit_errors = np.abs(padded_transmission - padded_reception)

        return BitErrorEvaluation(bit_errors)

    @property
    def abbreviation(self) -> str:
        return "BER"

    @property
    def title(self) -> str:
        return "Bit Error Rate Evaluation"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)


class BlockErrorArtifact(ArtifactTemplate[np.float64]):
    """Artifact of a block error evaluation between two modems exchanging information."""

    ...  # pragma: no cover


class BlockErrorEvaluation(ErrorEvaluation):
    """Block error evaluation of a single communication process between modems."""

    @property
    def title(self) -> str:
        return "Block Error Evaluation"

    @property
    def _x_axis_label(self) -> str:
        return "Block Index"

    @property
    def _y_axis_label(self) -> str:
        return "Block Error Indicator"

    def artifact(self) -> BlockErrorArtifact:
        bler = np.mean(self.evaluation)
        return BlockErrorArtifact(bler)


class BlockErrorEvaluator(CommunicationEvaluator, Serializable):
    """Evaluate block errors between two modems exchanging information."""

    def __init__(
        self,
        transmitting_modem: TransmittingModem,
        receiving_modem: ReceivingModem,
        plot_surface: bool = True,
    ) -> None:
        """
        Args:

            transmitting_modem (TransmittingModem):
                Modem transmitting information.

            receiving_modem (ReceivingModem):
                Modem receiving information.

            plot_surface (bool, optional):
                Plot the surface of the evaluation result in two-dimensional grids.
                Defaults to True.
        """

        CommunicationEvaluator.__init__(self, transmitting_modem, receiving_modem, plot_surface)
        self.plot_scale = "log"  # Plot logarithmically by default

    def evaluate(self) -> BlockErrorEvaluation:
        # Retrieve transmittend and received data
        transmission, reception = self._fetch_dsp_results()

        # Compare the decoded bit streams of each communication frame partioned into blocks into blocks
        # Every block with at least one bit error is considered a block error and increases the error counter
        num_tx_blocks = 0
        for tx_frame in transmission.frames:
            num_tx_blocks += tx_frame.bits.size // tx_frame.bit_block_size
        num_rx_blocks = 0
        for rx_frame in reception.frames:
            num_rx_blocks += rx_frame.decoded_bits.size // rx_frame.bit_block_size

        block_errors = np.ones(max(num_tx_blocks, num_rx_blocks), dtype=bool)
        b = 0
        for tx_frame, rx_frame in zip(transmission.frames, reception.frames):
            tx_blocks = tx_frame.bits.reshape((-1, tx_frame.bit_block_size))
            rx_blocks = rx_frame.decoded_bits.reshape((-1, rx_frame.bit_block_size))
            num_blocks = max(tx_blocks.shape[0], rx_blocks.shape[0])

            block_errors[b : b + num_blocks] = np.any(tx_blocks != rx_blocks, axis=1)
            b += num_blocks

        return BlockErrorEvaluation(block_errors)

    @property
    def title(self) -> str:
        return "Block Error Rate"

    @property
    def abbreviation(self) -> str:
        return "BLER"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)


class FrameErrorArtifact(ArtifactTemplate[float]):
    """Artifact of a frame error evaluation between two modems exchanging information."""

    ...  # pragma: no cover


class FrameErrorEvaluation(ErrorEvaluation):
    """Frame error evaluation of a single communication process between modems."""

    @property
    def title(self) -> str:
        return "Frame Error Evaluation"

    @property
    def _x_axis_label(self) -> str:
        return "Frame Index"

    @property
    def _y_axis_label(self) -> str:
        return "Frame Error Indicator"

    def artifact(self) -> FrameErrorArtifact:
        bler = float(np.mean(self.evaluation))
        return FrameErrorArtifact(bler)


class FrameErrorEvaluator(CommunicationEvaluator, Serializable):
    """Evaluate frame errors between two modems exchanging information."""

    def __init__(
        self,
        transmitting_modem: TransmittingModem,
        receiving_modem: ReceivingModem,
        plot_surface: bool = True,
    ) -> None:
        """
        Args:

            transmitting_modem (TransmittingModem):
                Modem transmitting information.

            receiving_modem (ReceivingModem):
                Modem receiving information.

            plot_surface (bool, optional):
                Plot the surface of the evaluation result in two-dimensional grids.
                Defaults to True.
        """

        CommunicationEvaluator.__init__(self, transmitting_modem, receiving_modem, plot_surface)
        self.plot_scale = "log"  # Plot logarithmically by default

    def evaluate(self) -> FrameErrorEvaluation:
        # Retrieve transmitted and received information
        transmission, reception = self._fetch_dsp_results()

        # The initial number of frame errors is the difference in transmitted and received frames,
        # since every dropped frame is considered an error
        frame_errors = np.ones(max(transmission.num_frames, reception.num_frames), dtype=bool)

        # Compare the decoded bit streams of each communication frame
        # If they differ, increase the frame errror count
        for f, (tx_frame, rx_frame) in enumerate(zip(transmission.frames, reception.frames)):
            frame_errors[f] = not np.array_equiv(tx_frame.bits, rx_frame.decoded_bits)

        return FrameErrorEvaluation(frame_errors)

    @property
    def title(self) -> str:
        return "Frame Error Rate"

    @property
    def abbreviation(self) -> str:
        return "FER"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)


class ThroughputArtifact(ArtifactTemplate[float]):
    """Artifact of a throughput evaluation between two modems exchanging information."""

    ...  # pragma: no cover


class ThroughputEvaluation(EvaluationTemplate[float, PlotVisualization]):
    """Throughput evaluation between two modems exchanging information."""

    def __init__(
        self, bits_per_frame: int, frame_duration: float, frame_errors: np.ndarray
    ) -> None:
        """
        Args:

            bits_per_frame (int):
                Number of bits per communication frame

            frame_duration (float):
                Duration of a single communication frame in seconds

            frame_errors (numpy.ndarray):
                Frame error indicators
        """

        num_frames = len(frame_errors)
        num_correct_frames = np.sum(np.invert(frame_errors))

        throughput = num_correct_frames * bits_per_frame / (num_frames * frame_duration)
        EvaluationTemplate.__init__(self, throughput)

    @property
    def title(self) -> str:
        return "Data Throughput"

    def artifact(self) -> ThroughputArtifact:
        return ThroughputArtifact(self.evaluation)

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> PlotVisualization:
        lines = np.empty_like(axes, dtype=np.object_)
        return PlotVisualization(figure, axes, lines)

    def _update_visualization(self, visualization: PlotVisualization, **kwargs) -> None:
        pass


class ThroughputEvaluator(CommunicationEvaluator, Serializable):
    """Evaluate data throughput between two modems exchanging information."""

    __framer_error_evaluator: FrameErrorEvaluator

    def __init__(
        self,
        transmitting_modem: TransmittingModem,
        receiving_modem: ReceivingModem,
        plot_surface: bool = True,
    ) -> None:
        """
        Args:

            transmitting_modem (TransmittingModem):
                Modem transmitting information.

            receiving_modem (ReceivingModem):
                Modem receiving information.

            plot_surface (bool, optional):
                Plot the surface of the evaluation result in two-dimensional grids.
                Defaults to True.
        """

        # Initialize base class
        CommunicationEvaluator.__init__(self, transmitting_modem, receiving_modem, plot_surface)

        # Initialize class attributes
        self.__framer_error_evaluator = FrameErrorEvaluator(transmitting_modem, receiving_modem)

    def evaluate(self) -> ThroughputEvaluation:
        # Get the frame errors
        frame_errors = self.__framer_error_evaluator.evaluate().evaluation.flatten()
        _, reception = self._fetch_dsp_results()

        # Transform frame errors to data throughput
        bits_per_frame = reception.frames[0].decoded_bits.size
        frame_duration = reception.frames[0].signal.duration

        return ThroughputEvaluation(bits_per_frame, frame_duration, frame_errors)

    @property
    def title(self) -> str:
        return "Data Throughput"

    @property
    def abbreviation(self) -> str:
        return "DRX"


class EVMArtifact(ArtifactTemplate[float]):
    """Artifact of a error vector magnitude (EVM) evaluation between two modems exchanging information."""

    ...  # pragma: no cover


class EVMEvaluation(EvaluationTemplate[float, PlotVisualization]):
    __transmitted_symbols: np.ndarray
    __received_symbols: np.ndarray
    __evm: float

    def __init__(self, transmitted_symbols: np.ndarray, received_symbols: np.ndarray) -> None:
        """
        Args:

            transmitted_symbols (numpy.ndarray): Originally transmitted communication symbols.
            received_symbols (numpy.ndarray): Received communication symbols.
        """

        _transmitted_symbols = transmitted_symbols.flatten()
        _received_symbols = received_symbols.flatten()
        size = min(transmitted_symbols.size, received_symbols.size)
        self.__transmitted_symbols = _transmitted_symbols[:size]
        self.__received_symbols = _received_symbols[:size]

        self.__evm = np.sqrt(
            np.mean(np.abs(self.__transmitted_symbols[:size] - self.__received_symbols[:size]) ** 2)
        )

    @property
    def title(self) -> str:
        return "Error Vector Magnitude"

    @property
    def abbreviation(self) -> str:
        return "EVM"

    def artifact(self) -> EVMArtifact:
        return EVMArtifact(self.__evm)

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> PlotVisualization:
        lines = np.empty_like(axes, dtype=np.object_)
        return PlotVisualization(figure, axes, lines)

    def _update_visualization(self, visualization: PlotVisualization, **kwargs) -> None:
        pass


class ConstellationEVM(CommunicationEvaluator):
    """Evaluate the error vector magnitude (EVM) of a constellation diagram."""

    def evaluate(self) -> EVMEvaluation:
        # Retrieve transmitted and received symbols
        transmission, reception = self._fetch_dsp_results()
        return EVMEvaluation(transmission.symbols.raw, reception.equalized_symbols.raw)

    @property
    def title(self) -> str:
        return "Error Vector Magnitude"

    @property
    def abbreviation(self) -> str:
        return "EVM"
