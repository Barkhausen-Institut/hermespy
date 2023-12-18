===================
Transmitting Modem
===================

.. inheritance-diagram:: hermespy.modem.modem.TransmittingModem
   :parts: 1
   :top-classes: hermespy.core.device.Transmitter, hermespy.modem.modem.BaseModem

Transmitting modems represent the digital signal processing operations
performed within a communication system up to the point of
digital-to-analog conversion.

After a :class:`TransmittingModem<hermespy.modem.modem.TransmittingModem>` is added as a
type of :class:`Transmitter<hermespy.core.device.Transmitter>` to a :class:`Device<hermespy.core.device.Device>`,
a call to :meth:`Device.transmit<hermespy.core.device.Device.transmit>` will be delegated
to :meth:`TransmittingModem._transmit()<hermespy.modem.modem.TransmittingModem._transmit>`:

.. mermaid::

   sequenceDiagram

   Device ->>+ TransmittingModem: _transmit()

   loop Frame Generation

   TransmittingModem->>+BitsSource: generate_bits()
   BitsSource->>-TransmittingModem: bits
   
   TransmittingModem->>+EncoderManager: encode()
   EncoderManager->>-TransmittingModem: encoded_bits

   TransmittingModem->>+CommunicationWaveform: map()
   CommunicationWaveform->>-TransmittingModem: Symbols

   TransmittingModem->>+SymbolPrecoding: encode()
   SymbolPrecoding->>-TransmittingModem: Symbols

   TransmittingModem->>+CommunicationWaveform: place()
   CommunicationWaveform->>-TransmittingModem: Symbols

   TransmittingModem->>+CommunicationWaveform: modulate()
   CommunicationWaveform->>-TransmittingModem: Signal

   TransmittingModem->>+TransmitStreamCoding: encode()
   TransmitStreamCoding->>-TransmittingModem: Signal

   end

   TransmittingModem ->>- Device: CommunicationTransmission

Initially, the :class:`TransmittingModem<hermespy.modem.modem.TransmittingModem>` configured :class:`BitsSource<hermespy.modem.bits_source.BitsSource>`
will be queried for a sequence of data bits required to moulate a single frame.
The specic number of bits depends several factors, primarily the :class:`Waveform<hermespy.modem.waveform.CommunicationWaveform>` configured by the respective
:attr:`waveform<hermespy.modem.modem.BaseModem.waveform>`, the precodings and the forward error correction.
The sequence of data bits is subseuently encoded for forward error correction by the :class:`EncoderManager<hermespy.fec.coding.EncoderManager>` configured by the respective
:attr:`encoder_manager<hermespy.modem.modem.BaseModem.encoder_manager>`.
The resulting sequence of encoded bits is then mapped to a sequence of communication symbols by the :class:`CommunicationWaveform's<hermespy.modem.waveform.CommunicationWaveform>`
:meth:`map<hermespy.modem.waveform.CommunicationWaveform.map>` method.
In the following step, the sequence of communication symbols is precoded by the :class:`SymbolPrecoding<hermespy.modem.symbol_precoding.SymbolPrecoding>` configured by the respective
:attr:`precoding<hermespy.modem.modem.BaseModem.precoding>` property.
Finally, a set of baseband streams is generate by :meth:`placing<hermespy.modem.waveform.CommunicationWaveform.place>` and :meth:`modulating<hermespy.modem.waveform.CommunicationWaveform.modulate>`
the precoded communication symbols.
The baseband streams are then encoded by the :attr:`TransmitStreamCoding<hermespy.modem.modem.TransmittingModem.transmit_stream_coding>`.
This sequence of steps can be repeated multiple times to generate a sequence of communication frames, if required.

Note that, as a bare minimum, only the :class:`waveform<hermespy.modem.modem.BaseModem.waveform>` has to be configured for a fully functional :class:`TransmittingModem<hermespy.modem.modem.TransmittingModem>`.
The following snippet shows how to configure a :class:`TransmittingModem<hermespy.modem.modem.TransmittingModem>` with a :class:`RootRaisedCosineWaveform<hermespy.modem.waveform_single_carrier.RootRaisedCosineWaveform>` wavform implementing a
:class:`CommunicationWaveform<hermespy.modem.waveform.CommunicationWaveform>`:

.. literalinclude:: ../scripts/examples/modem_TransmittingModem.py
   :language: python
   :linenos:
   :lines: 10-26

This barebone configuration can be extended by adding additional components such as
:class:`BitsSources<hermespy.modem.bits_source.BitsSource>`, :class:`Bit Encoders<hermespy.fec.coding.Encoder>`,
:class:`Symbol Precoders<hermespy.modem.precoding.symbol_precoding.SymbolPrecoding>` and
:class:`Stream Precoders<hermespy.precoding.stream_precoding.TransmitStreamEncoder>`:

.. literalinclude:: ../scripts/examples/modem_TransmittingModem.py
   :language: python
   :linenos:
   :lines: 28-39

.. autoclass:: hermespy.modem.modem.TransmittingModem
   :private-members: _transmit
   :show-inheritance:

.. footbibliography::
