# -*- coding: utf-8 -*-
"""
=====
Modem
=====

Within HermesPy, the Modem class represents the full signal processing chain configuration for the transmission
of information in form of bits.
Modems are a form of :class:`hermespy.core.device.DuplexOperator`,
binding to both the transmitter and receiver slot of :class:`hermespy.core.device.Device` objects.
In other words, they both transmit and receive complex waveforms over the devices.
It acts as a managing container for 5 signal processing step abstractions:

================================================= =======================================
Processing Step                                   Description
================================================= =======================================
:class:`.BitsSource`                              Source of data bits to be transmitted
:class:`hermespy.fec.EncoderManager`              Forward error correction configuration
:class:`.WaveformGenerator`                       Communication waveform configuration
:class:`hermespy.precoding.SymbolPrecoding`       Symbol precoding configuration
:class:`hermespy.precoding.TransmitStreamCoding`  Transmit MIMO configuration
:class:`hermespy.precoding.ReceiveStreamCoding`   Receive MIMO configuration
================================================= =======================================

.. mermaid::
   :caption: Modem Signal Processing Chain

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

       subgraph Modem
           direction LR

           subgraph BitSource

               direction TB
               Bits

           end

           subgraph BitCoding

               direction TB
               BitEncoding[Encoding]
               BitDecoding[Decoding]

           end

           subgraph Waveform

               Mapping --> Modulation
               ChannelEstimation[Channel Estimation]
               Synchronization
               Unmapping --- Demodulation
           end

           subgraph StreamCoding

               StreamEncoding[Encoding]
               StreamDecoding[Decoding]
           end

           subgraph BeamForming

               TxBeamform[Tx Beamforming]
               RxBeamform[Rx Beamforming]
           end

           Bits --> BitEncoding
           BitEncoding --> Mapping
           Modulation --> StreamEncoding
           StreamEncoding --> TxBeamform
           StreamDecoding --> RxBeamform
           Demodulation --- StreamDecoding
           Synchronization --- StreamDecoding
           ChannelEstimation --- StreamEncoding
           ChannelEstimation --- StreamDecoding
           BitDecoding --- Unmapping
       end

       subgraph Device

           direction TB
           txslot>Tx Slot]
           rxslot>Rx Slot]
       end

   txsignal{{Tx Signal Model}}
   txbits{{Tx Bits}}
   txsymbols{{Tx Symbols}}
   rxsignal{{Rx Signal Model}}
   rxbits{{Rx Bits}}
   rxsymbols{{Rx Symbols}}

   TxBeamform --> txsignal
   RxBeamform --> rxsignal
   txsignal --> txslot
   rxsignal --> rxslot

   Bits --> txbits
   Mapping --> txsymbols
   BitDecoding --> rxbits
   Unmapping --> rxsymbols

.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

       subgraph Modem

           direction LR

           subgraph BitSource

               direction TB

               Random([RandomSource]) === Stream([StreamSource])


           end

           subgraph BitCoding

                direction TB
                LDPC[/LDPC/]
                Interleaving[/Block-Interleaving/]
                CRC[/CRC/]
                Repetition[/Repetition/]
                Scrambler3G[/3GPP Scrambling/]
                Scrambler80211a[/802.11a Scrambling/]

                LDPC === Interleaving
                Interleaving === CRC
                CRC === Repetition
                Repetition === Scrambler3G
                Scrambler3G === Scrambler80211a

           end

           subgraph Waveform

                direction TB
                FSK([FSK])
                OFDM([OFDM])
                GFDM([GFDM])
                QAM([QAM])

                FSK === OFDM
                OFDM === GFDM
                GFDM === QAM
           end

           subgraph StreamCoding

              direction TB
              SC[/Single Carrier/]
              SM[/Spatial Multiplexing/]
              DFT[/DFT/]
              STBC[/STBC/]
              MRC[/Maximum Ratio Combining/]

              SC === SM === DFT === STBC === MRC

           end

           subgraph BeamForming

               direction TB
               Conventional[/Conventional/]
           end

            BitSource ===> BitCoding
            BitCoding <===> Waveform
            Waveform <===> StreamCoding
            StreamCoding <===> BeamForming
       end

    linkStyle 0 display: None
    linkStyle 1 display: None
    linkStyle 2 display: None
    linkStyle 3 display: None
    linkStyle 4 display: None
    linkStyle 5 display: None
    linkStyle 6 display: None
    linkStyle 7 display: None
    linkStyle 8 display: None
    linkStyle 9 display: None
    linkStyle 10 display: None
    linkStyle 11 display: None
    linkStyle 12 display: None
    linkStyle 11 display: None
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Set, Tuple, Type, Optional

import numpy as np
from h5py import Group

from hermespy.fec import EncoderManager
from hermespy.core import ChannelStateInformation, RandomNode, Transmission, Reception, Serializable, Signal, Device, Transmitter, Receiver, SNRType
from hermespy.precoding import SymbolPrecoding, ReceiveStreamCoding, TransmitStreamCoding
from .bits_source import BitsSource, RandomBitsSource
from .symbols import StatedSymbols, Symbols
from .waveform import WaveformGenerator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CommunicationTransmissionFrame(Transmission):
    """A single frame of information generated by transmitting over a communication operator."""

    signal: Signal
    """Communication base-band waveform."""

    bits: np.ndarray
    """Communication data bits."""

    encoded_bits: np.ndarray
    """Transmitted bits after FEC encoding."""

    symbols: Symbols
    """Communication data symbols."""

    encoded_symbols: Symbols
    """Communication data symbols after symbol encoding."""

    timestamp: float
    """Time at which the frame was transmitted in seconds."""

    def __init__(self, signal: Signal, bits: np.ndarray, encoded_bits: np.ndarray, symbols: Symbols, encoded_symbols: Symbols, timestamp: float) -> None:
        """
        Args:

            signal (Signal):
                Transmitted communication base-band waveform.

            bits (np.ndarray):
                Transmitted communication data bits.

            encoded_bits (np.ndarray):
                Transmitted communication bits after FEC encoding.

            symbols (Symbols):
                Transmitted communication data symbols.

            encoded_symbols (Symbols):
                Transmitted communication data symbols after symbol encoding.

            timestamp (float):
                Time at which the frame was transmitted in seconds.
        """

        Transmission.__init__(self, signal=signal)
        self.bits = bits
        self.encoded_bits = encoded_bits
        self.symbols = symbols
        self.encoded_symbols = encoded_symbols
        self.timestamp = timestamp

    @classmethod
    def from_HDF(cls: Type[CommunicationTransmissionFrame], group: Group, transmitter: Transmitter | None = None) -> CommunicationTransmissionFrame:
        # Recall groups
        signal = Signal.from_HDF(group["signal"])
        symbols = Symbols.from_HDF(group["symbols"])
        encoded_symbols = Symbols.from_HDF(group["encoded_symbols"])

        # Recall datasets
        bits = np.array(group["bits"], dtype=np.uint8)
        encoded_bits = np.array(group["encoded_bits"], dtype=np.uint8)

        # Recall attributes
        timestamp = group.attrs.get("timestamp", 0)

        # Initialize object from recalled state
        return cls(signal=signal, bits=bits, encoded_bits=encoded_bits, symbols=symbols, encoded_symbols=encoded_symbols, timestamp=timestamp)

    def to_HDF(self, group: Group) -> None:
        # Serialize base class
        self.signal.to_HDF(group.create_group("signal"))

        # Serialize groups
        self.symbols.to_HDF(group.create_group("symbols"))
        self.encoded_symbols.to_HDF(group.create_group("encoded_symbols"))

        # Serialize datasets
        group.create_dataset("bits", data=self.bits)
        group.create_dataset("encoded_bits", data=self.encoded_bits)

        # Serialize attributes
        group.attrs["timestamp"] = self.timestamp


class CommunicationTransmission(Transmission):
    """Information generated by transmitting over a communication operator."""

    frames: List[CommunicationTransmissionFrame]
    """Individual transmitted communication frames."""

    def __init__(self, signal: Signal, frames: Optional[List[CommunicationTransmissionFrame]] = None) -> None:
        """
        Args:

            signal (Signal):
                Transmitted communication base-band waveform.

            frames (List[CommunicationTransmissionFrame], optional):
                Individual transmitted communication frames.
        """

        Transmission.__init__(self, signal=signal)
        self.frames = [] if frames is None else frames

    @property
    def num_frames(self) -> int:
        """Number of transmitted communication frames.

        Returns:
            Number of frames.
        """

        return len(self.frames)

    @cached_property
    def bits(self) -> np.ndarray:
        """Transmitted bits before FEC encoding.

        Returns: Numpy array of transmitted bits.
        """

        concatenated_bits = np.empty(0, dtype=np.uint8)
        for frame in self.frames:
            concatenated_bits = np.append(concatenated_bits, frame.bits)

        return concatenated_bits

    @cached_property
    def symbols(self) -> Symbols:
        symbols = Symbols()
        for frame in self.frames:
            symbols.append_symbols(frame.symbols)

        return symbols

    @classmethod
    def from_HDF(cls: Type[CommunicationTransmission], group: Group) -> CommunicationTransmission:
        # Recall base signal
        signal = Signal.from_HDF(group["signal"])

        # Recall individual detected frames
        num_frames = group.attrs.get("num_frames", 0)
        frames = [CommunicationTransmissionFrame.from_HDF(group[f"frame_{f:02d}"]) for f in range(num_frames)]

        return cls(signal=signal, frames=frames)

    def to_HDF(self, group: Group) -> None:
        # Serialize attributes
        group.attrs["num_frames"] = self.num_frames

        # Serialize base information
        self.signal.to_HDF(group.create_group("signal"))

        # Serialize detected frames
        for f, frame in enumerate(self.frames):
            frame.to_HDF(group.create_group(f"frame_{f:02d}"))


class CommunicationReceptionFrame(object):
    """A single frame of information generated by receiving over a communication operator."""

    signal: Signal
    """Communication base-band waveform."""

    decoded_signal: Signal
    """Communication base-band waveform after MIMO stream decoding."""

    symbols: Symbols
    """Received communication symbols."""

    decoded_symbols: Symbols
    """Received communication symbols after precoding stage."""

    timestamp: float
    """Time at which the frame was transmitted in seconds."""

    equalized_symbols: Symbols
    """Equalized communication symbols."""

    encoded_bits: np.ndarray
    """Received encoded data bits before error correction."""

    decoded_bits: np.ndarray
    """Received decoded data bits after error correction."""

    csi: ChannelStateInformation
    """Estimated channel state."""

    def __init__(self, signal: Signal, decoded_signal: Signal, symbols: Symbols, decoded_symbols: Symbols, timestamp: float, equalized_symbols: Symbols, encoded_bits: np.ndarray, decoded_bits: np.ndarray, csi: ChannelStateInformation) -> None:
        self.signal = signal
        self.decoded_signal = decoded_signal
        self.symbols = symbols
        self.decoded_symbols = decoded_symbols
        self.timestamp = timestamp
        self.equalized_symbols = equalized_symbols
        self.encoded_bits = encoded_bits
        self.decoded_bits = decoded_bits
        self.csi = csi

    @classmethod
    def from_HDF(cls: Type[CommunicationReceptionFrame], group: Group) -> CommunicationReceptionFrame:
        # Recall groups
        signal = Signal.from_HDF(group["signal"])
        decoded_signal = Signal.from_HDF(group["decoded_signal"])
        symbols = Symbols.from_HDF(group["symbols"])
        decoded_symbols = Symbols.from_HDF(group["decoded_symbols"])
        equalized_symbols = Symbols.from_HDF(group["equalized_symbols"])
        csi = ChannelStateInformation.from_HDF(group["csi"])

        # Recall datasets
        encoded_bits = np.array(group["encoded_bits"], dtype=np.uint8)
        decoded_bits = np.array(group["decoded_bits"], dtype=np.uint8)

        # Recall attributes
        timestamp = group.attrs.get("timestamp", 0)

        # Initialize object from recalled state
        return cls(signal=signal, decoded_signal=decoded_signal, symbols=symbols, decoded_symbols=decoded_symbols, equalized_symbols=equalized_symbols, csi=csi, encoded_bits=encoded_bits, decoded_bits=decoded_bits, timestamp=timestamp)

    def to_HDF(self, group: Group) -> None:
        # Serialize groups
        self.signal.to_HDF(group.create_group("signal"))
        self.decoded_signal.to_HDF(group.create_group("decoded_signal"))
        self.symbols.to_HDF(group.create_group("symbols"))
        self.decoded_symbols.to_HDF(group.create_group("decoded_symbols"))
        self.equalized_symbols.to_HDF(group.create_group("equalized_symbols"))
        self.csi.to_HDF(group.create_group("csi"))

        # Serialize datasets
        group.create_dataset("encoded_bits", data=self.encoded_bits)
        group.create_dataset("decoded_bits", data=self.decoded_bits)

        # Serialize attributes
        group.attrs["timestamp"] = self.timestamp


class CommunicationReception(Reception):
    """Information generated by receiving over a communication operator."""

    frames: List[CommunicationReceptionFrame]
    """Individual received communication frames."""

    def __init__(self, signal: Signal, frames: Optional[List[CommunicationReceptionFrame]] = None) -> None:
        """
        Args:

            signal (Signal):
                Received communication base-band waveform.

            frames (List[CommunicationReceptionFrame], optional):
                Individual received communication frames.
        """

        Reception.__init__(self, signal=signal)
        self.frames = [] if frames is None else frames

    @property
    def num_frames(self) -> int:
        """Number of received communication frames.

        Returns:
            Number of frames.
        """

        return len(self.frames)

    @cached_property
    def encoded_bits(self) -> np.ndarray:
        """Received bits before FEC decoding.

        Returns:

            Numpy array containing received bits.
        """

        concatenated_bits = np.empty(0, dtype=np.uint8)
        for frame in self.frames:
            concatenated_bits = np.append(concatenated_bits, frame.encoded_bits)

        return concatenated_bits

    @cached_property
    def bits(self) -> np.ndarray:
        """Received bits after FEC decoding.

        Returns:

            Numpy array containing received bits.
        """

        concatenated_bits = np.empty(0, dtype=np.uint8)
        for frame in self.frames:
            concatenated_bits = np.append(concatenated_bits, frame.decoded_bits)

        return concatenated_bits

    @cached_property
    def symbols(self) -> Symbols:
        """Collect all communication symbols."""

        symbols = Symbols()
        for frame in self.frames:
            symbols.append_symbols(frame.symbols)

        return symbols

    @cached_property
    def equalized_symbols(self) -> Symbols:
        """Collect all equalized communication symbols."""

        symbols = Symbols()
        for frame in self.frames:
            symbols.append_symbols(frame.equalized_symbols)

        return symbols

    @classmethod
    def from_HDF(cls: Type[CommunicationReception], group: Group) -> CommunicationReception:
        # Recall base signal
        signal = Signal.from_HDF(group["signal"])

        # Recall individual detected frames
        num_frames = group.attrs.get("num_frames", 0)
        frames = [CommunicationReceptionFrame.from_HDF(group[f"frame_{f:02d}"]) for f in range(num_frames)]

        return cls(signal=signal, frames=frames)

    def to_HDF(self, group: Group) -> None:
        # Serialize attributes
        group.attrs["num_frames"] = self.num_frames

        # Serialize base information
        self.signal.to_HDF(group.create_group("signal"))

        # Serialize detected frames
        for f, frame in enumerate(self.frames):
            frame.to_HDF(group.create_group(f"frame_{f:02d}"))


class BaseModem(RandomNode, ABC):
    """Base class for wireless modems transmitting or receiving information over devices"""

    __encoder_manager: EncoderManager
    __precoding: SymbolPrecoding
    __waveform_generator: Optional[WaveformGenerator]

    @staticmethod
    def _arg_signature() -> Set[str]:
        return {"encoding", "precoding", "waveform", "seed"}

    def __init__(self, encoding: Optional[EncoderManager] = None, precoding: Optional[SymbolPrecoding] = None, waveform: Optional[WaveformGenerator] = None, seed: Optional[int] = None) -> None:
        """
        Args:

            encoding (EncoderManager, optional):
                Bit coding configuration.
                Encodes communication bit frames during transmission and decodes them during reception.

            precoding (SymbolPrecoding, optional):
                Modulation symbol coding configuration.

            waveform (WaveformGenerator, optional):
                The waveform to be transmitted by this modem.

            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.
        """

        # Base class initialization
        RandomNode.__init__(self, seed=seed)

        self.encoder_manager = EncoderManager() if encoding is None else encoding
        self.precoding = SymbolPrecoding(modem=self) if precoding is None else precoding
        self.waveform_generator = waveform

    @property
    @abstractmethod
    def transmitting_device(self) -> Optional[Device]:
        """Tranmsitting device operated by the modem.

        Returns:

            Handle to the transmitting device.
            `None` if the device is unspecified.
        """
        ...  # pragma no cover

    @property
    @abstractmethod
    def receiving_device(self) -> Optional[Device]:
        """Receiving device operated by the modem.

        Returns:

            Handle to the receiving device.
            `None` if the device is unspecified.
        """
        ...  # pragma no cover

    @property
    def num_transmit_streams(self) -> int:
        """The number of data streams handled by the modem during transmission.

        The number of data streams is always less or equal to the number of available antennas `num_antennas`.

        Returns:
            int:
                The number of data streams generated by the modem.
        """

        if self.transmitting_device is None:
            return 0

        # For now, stream compression will not be supported
        return self.transmitting_device.num_antennas

    @property
    def num_receive_streams(self) -> int:
        """The number of data streams handled by the modem during reception.

        The number of data streams is always less or equal to the number of available antennas `num_antennas`.

        Returns:
            int:
                The number of data streams processed by the modem.
        """

        if self.receiving_device is None:
            return 0

        # For now, stream compression will not be supported
        return self.receiving_device.num_antennas

    @property
    def encoder_manager(self) -> EncoderManager:
        """Access the modem's encoder management.

        Returns:
            EncoderManager:
                Handle to the modem's encoder instance.
        """

        return self.__encoder_manager

    @encoder_manager.setter
    def encoder_manager(self, new_manager: EncoderManager) -> None:
        """Update the modem's encoder management.

        Args:
            new_manager (EncoderManger):
                The new encoder manager.
        """

        self.__encoder_manager = new_manager
        self.__encoder_manager.random_mother = self
        new_manager.modem = self

    @property
    def waveform_generator(self) -> WaveformGenerator:
        """Communication waveform emitted by this modem.

        Returns:
            WaveformGenerator:
                Handle to the modem's `WaveformGenerator` instance.
        """

        return self.__waveform_generator

    @waveform_generator.setter
    def waveform_generator(self, value: Optional[WaveformGenerator]) -> None:
        """Set the communication waveform emitted by this modem."""

        self.__waveform_generator = value

        if value is not None:
            value.modem = self
            value.random_mother = self

    @property
    def precoding(self) -> SymbolPrecoding:
        """Access this modem's precoding configuration.

        Returns:
            SymbolPrecoding: Handle to the configuration.
        """

        return self.__precoding

    @precoding.setter
    def precoding(self, coding: SymbolPrecoding) -> None:
        """Modify the modem's precoding configuration.

        Args:
            coding (SymbolPrecoding): The new precoding configuration.
        """

        self.__precoding = coding
        self.__precoding.modem = self

    @property
    def num_data_bits_per_frame(self) -> int:
        """Compute the number of required data bits to generate a single frame.

        Returns:
            int: The number of data bits.
        """

        num_code_bits = self.waveform_generator.bits_per_frame
        return self.encoder_manager.required_num_data_bits(num_code_bits)

    @property
    def frame_duration(self) -> float:
        return self.waveform_generator.frame_duration

    @property
    def sampling_rate(self) -> float:
        return self.waveform_generator.sampling_rate

    def _noise_power(self, strength: float, snr_type: SNRType) -> float:
        # No waveform configured equals no noise required
        if self.waveform_generator is None:
            return 0.0

        if snr_type == SNRType.EBN0:
            return self.waveform_generator.bit_energy / strength

        if snr_type == SNRType.ESN0:
            return self.waveform_generator.symbol_energy / strength

        if snr_type == SNRType.PN0:
            return self.waveform_generator.power / strength

        raise ValueError(f"SNR of type '{snr_type}' is not supported by modem operators")

    @property
    def csi(self) -> Optional[ChannelStateInformation]:
        return None


class TransmittingModem(BaseModem, Transmitter[CommunicationTransmission], Serializable):
    """Representation of a wireless modem exclusively transmitting."""

    yaml_tag = "TxModem"

    # Source configuration of the transmitted bits
    __bits_source: BitsSource
    # Stream MIMO coding configuration
    __transmit_stream_coding: TransmitStreamCoding

    def __init__(self, bits_source: BitsSource | None = None, device: Device | None = None, *args, **kwargs) -> None:
        """
        Args:

            bits_source (BitsSource, optional):
                Source configuration of communication bits transmitted by this modem.
                Bits are randomly generated by default.
        """

        # Initialize base classes
        BaseModem.__init__(self, *args, **kwargs)
        Transmitter.__init__(self)

        # Initialize clas attributes
        self.bits_source = RandomBitsSource() if bits_source is None else bits_source
        self.__transmit_stream_coding = TransmitStreamCoding(modem=self)

        if device is not None:
            self.device = device

    @property
    def transmitting_device(self) -> Optional[Device]:
        # The transmitting device resolves to the operated device
        return self.device

    @property
    def receiving_device(self) -> Optional[Device]:
        return None

    @Transmitter.device.setter  # type: ignore
    def device(self, value: Device) -> None:
        if value is self.device:
            return

        if Transmitter.device.fget(self) is not None:  # type: ignore
            self.device.transmitters.remove(self)

        if value is not None and self not in value.transmitters:
            value.transmitters.add(self)

    @property
    def bits_source(self) -> BitsSource:
        """Source configuration of communication transmitted by this modem.

        Returns: A handle to the source configuration.
        """

        return self.__bits_source

    @bits_source.setter
    def bits_source(self, value: BitsSource):
        value.random_mother = self
        self.__bits_source = value

    @property
    def transmit_stream_coding(self) -> TransmitStreamCoding:
        """Stream MIMO coding configuration during signal transmission.

        Returns: Handle to the coding configuration.
        """

        return self.__transmit_stream_coding

    @transmit_stream_coding.setter
    def transmit_stream_coding(self, value: TransmitStreamCoding) -> None:
        self.__transmit_stream_coding = value
        value.modem = self

    def __map(self, bits: np.ndarray, num_streams: int) -> Symbols:
        """Map a block of information bits to commuication symbols.

        Args:

             bits (np.ndarray):
                The information bits to be mapped.

        Returns:
            A series of communication symbols.
        """

        symbols = Symbols()
        for frame_bits in np.reshape(bits, (num_streams, -1)):
            symbols.append_stream(self.waveform_generator.map(frame_bits))

        return symbols

    def __modulate(self, symbols: Symbols) -> Signal:
        """Modulates a sequence of MIMO signals into a base-band communication waveform.

        Args:

            symbols (Symbols):
                Communication symbols to be modulated.

        Returns:

            The modualted base-band communication frame.
        """

        signal = Signal.empty(self.waveform_generator.sampling_rate)

        if symbols.num_streams < 1:
            signal.append_streams(self.waveform_generator.modulate(Symbols()))

        else:
            for symbol_stream in symbols.raw:
                signal.append_streams(self.waveform_generator.modulate(Symbols(symbol_stream[np.newaxis, :, :])))

        return signal

    def _transmit(self, duration: float = -1.0) -> CommunicationTransmission:
        """Returns an array with the complex base-band samples of a waveform generator.

        The signal may be distorted by RF impairments.

        Args:
            duration (float, optional): Length of signal in seconds.

        Returns:

            Transmitted information.
        """

        # By default, the drop duration will be exactly one frame
        if duration <= 0.0:
            duration = self.frame_duration

        # Infer required parameters
        frame_duration = self.frame_duration
        num_mimo_frames = int(duration / frame_duration)
        code_bits_per_mimo_frame = int(self.waveform_generator.bits_per_frame * self.precoding.num_input_streams)
        data_bits_per_mimo_frame = self.encoder_manager.required_num_data_bits(code_bits_per_mimo_frame)

        if len(self.transmit_stream_coding) > 0:
            num_output_streams = self.transmit_stream_coding.num_output_streams

        elif len(self.precoding) > 0:
            num_output_streams = self.precoding.num_output_streams

        else:
            num_output_streams = 1

        # Assert that the number of output streams matches the antenna count
        if self.device is not None and num_output_streams != self.device.num_antennas:
            raise ValueError(f"Modem MIMO configuration generates invalid number of antenna streams ({num_output_streams} instead of {self.device.num_antennas})")

        signal = Signal.empty(self.waveform_generator.sampling_rate, num_output_streams)

        # Abort if no frame is to be transmitted within the current duration
        if num_mimo_frames < 1:
            transmission = CommunicationTransmission(signal)
            return transmission

        frames: List[CommunicationTransmissionFrame] = []
        for n in range(num_mimo_frames):
            # Generate plain data bits
            data_bits = self.bits_source.generate_bits(data_bits_per_mimo_frame)

            # Apply forward error correction
            encoded_bits = self.encoder_manager.encode(data_bits, code_bits_per_mimo_frame)

            # Map bits to communication symbols
            symbols = self.__map(encoded_bits, self.precoding.num_input_streams)

            # Apply precoding cofiguration
            encoded_symbols = self.precoding.encode(StatedSymbols(symbols.raw, np.ones((symbols.num_streams, 1, symbols.num_blocks, symbols.num_symbols), dtype=complex)))

            # Modulate to base-band signal representation
            frame_signal = self.__modulate(encoded_symbols)

            # Apply the stream transmit coding configuration
            encoded_frame_signal = self.__transmit_stream_coding.encode(frame_signal)

            # Save results
            signal.append_samples(encoded_frame_signal)
            frames.append(CommunicationTransmissionFrame(signal=frame_signal, bits=data_bits, encoded_bits=encoded_bits, symbols=symbols, encoded_symbols=encoded_symbols, timestamp=n * frame_duration))

        # Update the assumed signal carrier frequency to RF band
        signal.carrier_frequency = self.carrier_frequency

        # Save the transmitted information
        transmission = CommunicationTransmission(signal, frames)
        return transmission

    def _recall_transmission(self, group: Group) -> CommunicationTransmission:
        return CommunicationTransmission.from_HDF(group)


class ReceivingModem(BaseModem, Receiver[CommunicationReception], Serializable):
    """Representation of a wireless modem exclusively receiving."""

    yaml_tag = "RxModem"
    """YAML serialization tag"""

    # MIMO stream configuration during signal rececption
    __receive_stream_coding: ReceiveStreamCoding

    def __init__(self, device: Device | None = None, *args, **kwargs) -> None:
        self.__receive_stream_coding = ReceiveStreamCoding(modem=self)

        BaseModem.__init__(self, *args, **kwargs)
        Receiver.__init__(self)

        if device is not None:
            self.device = device

    @property
    def transmitting_device(self) -> Optional[Device]:
        return None

    @property
    def receiving_device(self) -> Optional[Device]:
        # The receiving device resolves to the operated device
        return self.device

    @Receiver.device.setter  # type: ignore
    def device(self, value: Device) -> None:
        if value is not None and self not in value.receivers:
            value.receivers.add(self)

        if value is self.device:
            return

        if Receiver.device.fget(self) is not None:  # type: ignore
            self.device.receivers.remove(self)

    @property
    def receive_stream_coding(self) -> ReceiveStreamCoding:
        """Stream MIMO coding configuration during signal reception.

        Returns: Handle to the coding configuration.
        """

        return self.__receive_stream_coding

    @receive_stream_coding.setter  # type: ignore
    def transmit_stream_coding(self, value: ReceiveStreamCoding) -> None:
        self.__receive_stream_coding = value
        value.modem = self

    def __synchronize(self, received_signal: Signal) -> Tuple[List[int], List[Signal]]:
        """Synchronize a received MIMO base-band stream.

        Converts the stream into sections representing communication frames.

        Args:

            received_signal (Signal):
                The MIMO signal received over the operated device's RF chain.

        Returns:

            A sequence signals representing communication frames and their respective detection indices.
        """

        # Synchronize raw MIMO data into frames
        frame_start_indices = self.waveform_generator.synchronization.synchronize(received_signal.samples)
        frame_length = self.waveform_generator.samples_in_frame

        synchronized_signals = []
        for frame_start in frame_start_indices:
            frame_stop = frame_start + frame_length

            if frame_stop <= received_signal.num_samples:
                frame_samples = received_signal.samples[:, frame_start:frame_stop]

            else:
                frame_samples = np.append(received_signal.samples[:, frame_start:], np.zeros((received_signal.num_streams, frame_stop - received_signal.num_samples), dtype=complex))

            frame_signal = Signal(frame_samples, received_signal.sampling_rate)
            synchronized_signals.append(frame_signal)

        return frame_start_indices, synchronized_signals

    def __demodulate(self, frame: Signal) -> Symbols:
        """Demodulates a sequence of synchronized MIMO signals into data symbols.

        Args:

            frame (Signal):
                Synchronized MIMO signal, representing the samples of a full communication frame.

        Returns:

            Demodulated frame symbols.
        """

        symbols = Symbols()
        for stream in frame.samples:
            stream_symbols = self.waveform_generator.demodulate(stream)
            symbols.append_stream(stream_symbols)

        return symbols

    def __unmap(self, symbols: Symbols) -> np.ndarray:
        """Unmap a set of communication symbols to information bits.

        Args:

             symbols (Symbols):
                The communication symbols to be unmapped

        Returns:
            A numpy array containing hard information bits.
        """

        bits = np.empty(0, dtype=np.uint8)
        for stream in symbols.raw:
            bits = np.append(bits, self.waveform_generator.unmap(Symbols(stream[np.newaxis, :, :])))

        return bits

    def _receive(self, signal: Signal, csi: ChannelStateInformation) -> CommunicationReception:
        # Resample the signal to match the waveform's requirements
        signal = signal.resample(self.waveform_generator.sampling_rate)

        # Synchronize incoming signals
        frame_start_indices, synchronized_signals = self.__synchronize(signal)

        # Abort if no frame has been detected
        if len(synchronized_signals) < 1:
            reception = CommunicationReception(signal)
            return reception

        # Infer required parameters
        code_bits_per_mimo_frame = int(self.waveform_generator.bits_per_frame * self.precoding.num_input_streams)
        data_bits_per_mimo_frame = self.encoder_manager.required_num_data_bits(code_bits_per_mimo_frame)

        # Process each frame independently
        frames: List[CommunicationReceptionFrame] = []
        for frame_index, frame_signal in zip(frame_start_indices, synchronized_signals):
            # Apply the stream transmit decoding configuration
            decoded_frame_signal = self.__receive_stream_coding.decode(frame_signal)

            # Demodulate raw symbols for each frame independtly
            symbols = self.__demodulate(decoded_frame_signal)

            # Estimate the channel from each frame demodulation
            stated_symbols, channel_estimate = self.waveform_generator.estimate_channel(symbols)

            # Decode the pre-equalization symbol precoding stage
            decoded_symbols = self.precoding.decode(stated_symbols)

            # Equalize the received symbols for each frame given the estimated channel state
            equalized_symbols = self.waveform_generator.equalize_symbols(decoded_symbols)

            # Unmap equalized symbols to information bits
            encoded_bits = self.__unmap(equalized_symbols)

            # Apply inverse FEC configuration to correct errors and remove redundancies
            decoded_bits = self.encoder_manager.decode(encoded_bits, data_bits_per_mimo_frame)

            # Store the received information
            frames.append(CommunicationReceptionFrame(signal=frame_signal, decoded_signal=decoded_frame_signal, symbols=symbols, decoded_symbols=decoded_symbols, timestamp=frame_index * signal.sampling_rate, equalized_symbols=equalized_symbols, encoded_bits=encoded_bits, decoded_bits=decoded_bits, csi=channel_estimate))

        # Store the received information of all frames
        reception = CommunicationReception(signal=signal, frames=frames)
        return reception

    def _recall_reception(self, group: Group) -> CommunicationReception:
        return CommunicationReception.from_HDF(group)

    @property
    def csi(self) -> Optional[ChannelStateInformation]:
        return Receiver.csi.fget(self)  # type: ignore


class DuplexModem(TransmittingModem, ReceivingModem):
    """Representation of a wireless modem simulatneously transmitting and receiving."""

    yaml_tag = "Modem"
    """YAML serialization tag"""

    def __init__(self, *args, **kwargs) -> None:
        ReceivingModem.__init__(self)
        TransmittingModem.__init__(self, *args, **kwargs)

    @property
    def transmitting_device(self) -> Optional[Device]:
        return TransmittingModem.transmitting_device.fget(self)  # type: ignore

    @property
    def receiving_device(self) -> Optional[Device]:
        return ReceivingModem.receiving_device.fget(self)  # type: ignore

    @TransmittingModem.device.setter  # type: ignore
    def device(self, value: Device) -> None:
        TransmittingModem.device.fset(self, value)
        ReceivingModem.device.fset(self, value)

    @property
    def csi(self) -> Optional[ChannelStateInformation]:
        return Receiver.csi.fget(self)  # type: ignore


class SimplexLink(TransmittingModem, ReceivingModem, Serializable):
    """Convenience class to manage a simplex communication link between two dedicated devices."""

    yaml_tag = "SimplexLink"
    """YAML serialization tag"""

    __transmitting_device: Device  # Transmitting device
    __receiving_device: Device  # Receiving device

    def __init__(self, transmitting_device: Device, receiving_device: Device, *args, **kwargs) -> None:
        """

            transmitting_device (Device):
                Transmitting device.

            receiving_device (Device):
                Receiving device.

            args, kwargs:
                Modem initialization parameters.
                Refer to :class:`.TransmittingModem` and :class:`.ReceivingModem` for further details.

        Raises:

            ValueError: If `transmitting_device` and `receiving_device` are identical.
        """

        # Make sure the link is established between two dedicated devices
        # if transmitting_device is receiving_device:
        #    raise ValueError("Transmitter and receiver must be two independent device instances")

        self.__transmitting_device = transmitting_device
        self.__receiving_device = receiving_device

        TransmittingModem.__init__(self, *args, **kwargs)
        ReceivingModem.__init__(self, *args, **kwargs)

        transmitting_device.transmitters.add(self)
        receiving_device.receivers.add(self)

        return

    @property
    def transmitting_device(self) -> Device:
        """The transmitting device.

        Returns: Device handle.
        """

        return self.__transmitting_device

    @ReceivingModem.reference.setter  # type: ignore
    def reference(self, _: Device) -> None:

        raise RuntimeError("Specifying the reference device of a simplex link is not supported")

    @ReceivingModem.reference.getter  # type: ignore
    def reference(self) -> Device:

        return self.transmitting_device

    @property
    def receiving_device(self) -> Device:
        """The receiving device.

        Returns: Device handle.
        """

        return self.__receiving_device
