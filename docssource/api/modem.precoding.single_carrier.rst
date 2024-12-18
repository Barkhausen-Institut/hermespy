=======================
Single Carrier Decoding
=======================

.. inheritance-diagram:: hermespy.modem.precoding.single_carrier.SingleCarrier
   :parts: 1
   :top-classes: hermespy.modem.precoding.symbol_precoding.SymbolPrecoder

Single carrier decoding combines individually demodulated symbols from multiple antenna streams
into a single stream of symbols, given that they have been transmitted by a single-antenna device.

It can be configured by adding an instance to the
:class:`SymbolPrecoding<hermespy.modem.precoding.symbol_precoding.ReceiveSymbolCoding>`
of a :class:`Modem<hermespy.modem.modem.ReceivingModem>` exposed by the :attr:`precoding<hermespy.modem.modem.ReceivingModem.receive_symbol_coding>`
attribute:

.. literalinclude:: ../scripts/examples/modem_precoding_sc.py
   :language: python
   :linenos:
   :lines: 19-43

Note that decoding requires channel state information at the receiver,
therefore waveform's :attr:`channel_estimation<hermespy.modem.waveform.CommunicationWaveform.channel_estimation>` attribute must be configured.

.. autoclass:: hermespy.modem.precoding.single_carrier.SingleCarrier

.. footbibliography::
