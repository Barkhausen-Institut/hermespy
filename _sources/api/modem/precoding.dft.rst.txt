===========================
Discrete Fourier Transform
===========================

.. inheritance-diagram:: hermespy.modem.precoding.dft.DFT
   :parts: 1
   :top-classes: hermespy.modem.precoding.symbol_precoding.SymbolPrecoder

Discrete Fourier precodings apply a Fourier transform to the input symbols
in between the mapping and modulation stage during transmission.
Inversely, the precoding is applied after demodulation and before the
demapping stage during reception.

They can be configured by adding an instance to the precoding configuration
of a :class:`Modem<hermespy.modem.modem.TransmittingModemBase>` or :class:`Modem<hermespy.modem.modem.ReceivingModemBase>` exposed by the :attr:`TransmittingModemBase.precoding<hermespy.modem.modem.TransmittingModemBase.transmit_symbol_coding>` / :attr:`ReceivingModemBase.precoding<hermespy.modem.modem.ReceivingModemBase.receive_symbol_coding>` attributes:

.. literalinclude:: ../../scripts/examples/modem_precoding_dft.py
   :language: python
   :linenos:
   :lines: 17-42

.. autoclass:: hermespy.modem.precoding.dft.DFT

.. autoclass:: hermespy.modem.precoding.dft.DFTNorm

.. footbibliography::
