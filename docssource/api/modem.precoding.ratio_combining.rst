=======================
Maximal Ratio Combining
=======================

.. inheritance-diagram:: hermespy.modem.precoding.ratio_combining.MaximumRatioCombining
   :parts: 1
   :top-classes: hermespy.modem.precoding.symbol_precoding.SymbolPrecoder

Maximal Ratio Combining combines individually demodulated symbols from multiple antenna streams
into a single stream of symbols in an SNR-optimal fashion, given that they have been transmitted by a single-antenna device.

It can be configured by adding an instance to the
:class:`SymbolPrecoding<hermespy.modem.precoding.symbol_precoding.SymbolPrecoding>`
of a :class:`Modem<hermespy.modem.modem.BaseModem>` exposed by the :attr:`precoding<hermespy.modem.modem.BaseModem.precoding>`
attribute:

.. literalinclude:: ../scripts/examples/modem_precoding_mrc.py
   :language: python
   :linenos:
   :lines: 19-43

Note that decoding requires channel state information at the receiver,
therefore waveform's :attr:`channel_estimation<hermespy.modem.waveform.CommunicationWaveform.channel_estimation>` attribute must be configured.

.. autoclass:: hermespy.modem.precoding.ratio_combining.MaximumRatioCombining

.. footbibliography::
