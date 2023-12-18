=======================
Space-Time Block Codes
=======================

.. inheritance-diagram:: hermespy.modem.precoding.space_time_block_coding.Alamouti hermespy.modem.precoding.space_time_block_coding.Ganesan
   :parts: 1
   :top-classes: hermespy.modem.precoding.symbol_precoding.SymbolPrecoder


Space-Time Block codes distribute communication symbols over multiple antennas and time slots.
They can be configured by adding an instance to the
:class:`SymbolPrecoding<hermespy.modem.precoding.symbol_precoding.SymbolPrecoding>`
of a :class:`Modem<hermespy.modem.modem.BaseModem>` exposed by the :attr:`precoding<hermespy.modem.modem.BaseModem.precoding>`
attribute.

The following example shows how to configure a modem with a Alamouti precoder within a :math:`2\times 2` MIMO communication link:

.. literalinclude:: ../scripts/examples/modem_precoding_alamouti.py
   :language: python
   :linenos:
   :lines: 19-45

Note that Alamouti precoding requires channel state information at the receiver,
therefore waveform's :attr:`channel_estimation<hermespy.modem.waveform.CommunicationWaveform.channel_estimation>` attribute must be configured.

The following example shows how to configure a modem with a Ganesan precoder within a :math:`4\times 4` MIMO communication link:

.. literalinclude:: ../scripts/examples/modem_precoding_ganesan.py
   :language: python
   :linenos:
   :lines: 19-45

Note that Ganesan precoding requires channel state information at the receiver,
therefore waveform's :attr:`channel_estimation<hermespy.modem.waveform.CommunicationWaveform.channel_estimation>` attribute must be configured.

.. autoclass:: hermespy.modem.precoding.space_time_block_coding.Alamouti

.. autoclass:: hermespy.modem.precoding.space_time_block_coding.Ganesan

.. footbibliography::
