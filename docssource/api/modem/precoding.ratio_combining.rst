=======================
Maximal Ratio Combining
=======================

.. inheritance-diagram:: hermespy.modem.precoding.ratio_combining.MaximumRatioCombining
   :parts: 1

Maximal Ratio Combining combines individually demodulated symbols from multiple antenna streams
into a single stream of symbols in an SNR-optimal fashion, given that they have been transmitted by a single-antenna device.

It can be configured by adding an instance to the
:class:`SymbolPrecoding<hermespy.modem.precoding.symbol_precoding.ReceiveSymbolDecoder>`
of a :class:`Modem<hermespy.modem.modem.ReceivingModemBase>` exposed by the :attr:`receive_symbol_coding<hermespy.modem.modem.ReceivingModemBase.receive_symbol_coding>`
attribute:

.. literalinclude:: ../../scripts/examples/modem_precoding_mrc.py
   :language: python
   :linenos:
   :lines: 19-43

Note that decoding requires channel state information at the receiver,
therefore waveform's :attr:`channel_estimation<hermespy.modem.waveform.CommunicationWaveform.channel_estimation>` attribute must be configured.

.. autoclass:: hermespy.modem.precoding.ratio_combining.MaximumRatioCombining

.. footbibliography::
