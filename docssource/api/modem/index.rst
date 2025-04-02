=============
Communication
=============

The communication package contains a propgramming framework for
simulating and evaluating communication systems on the physical layer.
It is primarily comprised of :doc:`Modem<modem.BaseModem>` implementations
and associated :doc:`Communication Waveforms<waveform>`:

.. mermaid::

   classDiagram

   class CommunicationWaveform {

      <<Abstract>>

      %%+int oversampling_factor
      %%+int modulation_order
      %%+int num_data_symbols*
      %%+int samples_per_frame*
      %%+float frame_duration
      %%+float symbol_duration*
      %%+float bit_energy*
      %%+float symbol_energy*
      %%+float power*
      +map(numpy.ndarray) Symbols*
      +unmap(Symbols) np.ndarray*
      +place(Symbols) Symbols*
      +pick(StatedSymbols) StatedSymbols*
      +modulate(Symbols) np.ndarray*
      +demodulate(np.ndarray) Symbols*
   }

   class BaseModem {

      <<Abstract>>

      +CommunicationWaveform waveform
      +EncoderManager encoder_manager
      +SymbolPredocding Precoding
      +Device transmitting_device*
      +Device receiving_device*
   }

   class TransmittingModem {

      +Device transmitting_device
      +None receiving_device
      +BitsSource bits_source
      +TransmitStreamCoding transmit_stream_coding
      #CommunicationTransmission _transmit()
   }

   class ReceivingModem {

      +None transmitting_device
      +Device receiving_device
      +BitsSink bits_sink
      +ReceiveStreamCoding receive_stream_coding
      #CommunicationReception _receive(Signal)
   }

   class SimplexLink
   class DuplexModem

   class CommunicationTransmission {

      +List[CommunicationTransmissionFrame] frames
      +int num_frames
      +bits numpy.ndarray
      +symbols Symbols
   }

   class CommunicationReception {

      +List[CommunicationReceptionFrame] frames
      +int num_frames
      +encoded_bits numpy.ndarray
      +bits numpy.ndarray
      +symbols Symbols
      +equalized_symbols Symbols
   }

   BaseModem --* CommunicationWaveform
   TransmittingModem --|> BaseModem
   TransmittingModem --> CommunicationTransmission : create
   ReceivingModem --|> BaseModem
   ReceivingModem --> CommunicationReception : create
   SimplexLink --|> TransmittingModem
   SimplexLink --|> ReceivingModem
   DuplexModem --|> TransmittingModem
   DuplexModem --|> ReceivingModem

   link CommunicationWaveform "waveform.waveform.html"
   link BaseModem "modem.BaseModem.html"
   link TransmittingModem "modem.TransmittingModem.html"
   link ReceivingModem "modem.ReceivingModem.html"
   link SimplexLink "modem.SimplexLink.html"
   link DuplexModem "modem.DuplexModem.html"
   link CommunicationTransmission "modem.modem.CommunicationTransmission.html"
   link CommunicationReception "modem.CommunicationReception.html"

:doc:`Modems<modem.BaseModem>` implement a customizable signal processing pipeline
for both :doc:`transmitting<modem.TransmittingModem>` and :doc:`receiving<modem.ReceivingModem>`
communication devices,
as well as a :doc:`SimplexLink<modem.SimplexLink>` for unidirectional communication links
and a :doc:`DuplexModem<modem.DuplexModem>` for bidirectional communication links.
They can be configured in terms of their :doc:`bit generation<bits_source>`
and the :doc:`forward error correction</api/fec/index>` codings applied to the bits,
the :doc:`precoding<precoding>` applied to communication symbols,
and, most importantly, the :doc:`communication waveform<waveform>` used to transmit and receive commuication symbols.
The :doc:`waveform<waveform>` offers additional specific configuration options for synchronization,
channel estimation and channel equalization.
The following waveform types are currently supported:

.. include:: waveform._table.rst

Out of the box, the communication package provides a number of evaluators to estimate common performance metrics
of communication systems:

.. include:: evaluators._table.rst

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   waveform
   evaluators
   precoding
   symbols
   bits_source
   modem.TransmittingModem
   modem.ReceivingModem
   modem.SimplexLink
   modem.DuplexModem
   modem.BaseModem
   modem.CommunicationTransmission
   modem.CommunicationTransmissionFrame
   modem.CommunicationReception
   modem.CommunicationReceptionFrame
   mapping
   frame_generator