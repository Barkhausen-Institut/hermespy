===================
Receiving Modem
===================

.. inheritance-diagram:: hermespy.modem.modem.ReceivingModem
   :parts: 1
   :top-classes: hermespy.core.device.Receiver, hermespy.modem.modem.BaseModem

Receiving modems represent the digital signal processing operations
performed within a communication system for the point of
analog-to-digital conversion up to the point of decoding the received bits.

After a :class:`ReceivingModem<hermespy.modem.modem.ReceivingModem>` is added as a
type of :class:`Receiver<hermespy.core.device.Receiver>` to a :class:`Device<hermespy.core.device.Device>`,
a call to :meth:`Device.receive<hermespy.core.device.Device.receive>` will be delegated
to :meth:`ReceivingModem._receive<hermespy.modem.modem.ReceivingModemBase._receive>`:

.. mermaid::

   sequenceDiagram

   Device ->>+ ReceivingModem: receive(Signal)

   ReceivingModem->>+CommunicationWaveform: synchronize(Signal)
   CommunicationWaveform->>-ReceivingModem: frame_indices

   loop Frame Reception

   ReceivingModem->>+ReceiveStreamCoding: decode(Signal)
   ReceiveStreamCoding->>-ReceivingModem: Signal

   ReceivingModem->>+CommunicationWaveform: demodulate(Signal)
   CommunicationWaveform->>-ReceivingModem: Symbols
   
   ReceivingModem->>+CommunicationWaveform: estimate_channel(Symbols)
   CommunicationWaveform->>-ReceivingModem: StatedSymbols

   ReceivingModem->>+CommunicationWaveform: pick(StatedSymbols)
   CommunicationWaveform->>-ReceivingModem: StatedSymbols

   ReceivingModem->>+SymbolPrecoding: decode(StatedSymbols)
   SymbolPrecoding->>-ReceivingModem: StatedSymbols

   ReceivingModem->>+CommunicationWaveform: equalize_symbols(StatedSymbols)
   CommunicationWaveform->>-ReceivingModem: Symbols

   ReceivingModem->>+CommunicationWaveform: unmap(StatedSymbols)
   CommunicationWaveform->>-ReceivingModem: Bits

   ReceivingModem->>+EncoderManager: decode(Bits)
   EncoderManager->>-ReceivingModem: Bits

   end

   ReceivingModem ->>- Device: CommunicationReception


Initially, the :class:`ReceivingModem<hermespy.modem.modem.ReceivingModem>` will synchronize incoming
:class:`Signals<hermespy.core.signal_model.Signal>`, partitionin them into individual frames.
For each frame, the :class:`ReceiveSignalCoding<hermespy.core.precoding.ReceiveSignalCoding>`
configured by the :attr:`receive_signal_coding<hermespy.modem.modem.ReceivingModemBase.receive_signal_coding>`
will be used to decode the incoming base-band sample streams from each :class:`AntennaPort<hermespy.core.antennas.AntennaPort>`.
Afterwards, each decoded stream will be :meth:`demodulated<hermespy.modem.waveform.CommunicationWaveform.demodulate>`,
the channel will be :meth:`estimated<hermespy.modem.waveform.CommunicationWaveform.estimate_channel>` and
the resulting :class:`StatedSymbols<hermespy.modem.symbols.StatedSymbols>` will be :meth:`picked<hermespy.modem.waveform.CommunicationWaveform.pick>`.
The :class:`StatedSymbols<hermespy.modem.symbols.StatedSymbols>` will then be :meth:`decoded<hermespy.modem.precoding.symbol_precoding.ReceiveSymbolCoding.decode_symbols>`
and :meth:`equalized<hermespy.modem.waveform.CommunicationWaveform.equalize_symbols>`.
Finally, the :class:`Symbols<hermespy.modem.symbols.Symbols>` will be :meth:`unmapped<hermespy.modem.waveform.CommunicationWaveform.unmap>`
and the error correction will be :meth:`decoded<hermespy.fec.coding.EncoderManager.decode>`.

Note that, as a bare minimum, only the :class:`waveform<hermespy.modem.modem.BaseModem.waveform>` has to be configured for a fully functional :class:`ReceivingModem<hermespy.modem.modem.ReceivingModem>`.
The following snippet shows how to configure a :class:`ReceivingModem<hermespy.modem.modem.ReceivingModem>` with a :class:`RootRaisedCosineWaveform<hermespy.modem.waveform_single_carrier.RootRaisedCosineWaveform>` wavform implementing a
:class:`CommunicationWaveform<hermespy.modem.waveform.CommunicationWaveform>`:

.. literalinclude:: ../../scripts/examples/modem_ReceivingModem.py
   :language: python
   :linenos:
   :lines: 19-34

The barebone configuration can be extend by additional components such as
:class:`Synchronization<hermespy.modem.waveform.Synchronization>`,
:class:`ReceiveSignalCoding<hermespy.core.precoding.ReceiveSignalCoding>`,
:class:`Channel Estimation<hermespy.modem.waveform.ChannelEstimation>`,
:class:`ReceiveSymbolCoding<hermespy.modem.precoding.symbol_precoding.ReceiveSymbolCoding>`,
:class:`Channel Equalization<hermespy.modem.waveform.ChannelEqualization>` and
:class:`Bit Encoders<hermespy.fec.coding.Encoder>`:

.. literalinclude:: ../../scripts/examples/modem_ReceivingModem.py
   :language: python
   :linenos:
   :lines: 36-56

.. autoclass:: hermespy.modem.modem.ReceivingModem

.. autoclass:: hermespy.modem.modem.ReceivingModemBase
      :private-members: _receive

.. footbibliography::
