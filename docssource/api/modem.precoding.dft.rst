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

They can be configured by adding an instance to the
:class:`SymbolPrecoding<hermespy.modem.precoding.symbol_precoding.SymbolPrecoding>`
of a :class:`Modem<hermespy.modem.modem.BaseModem>` exposed by the :attr:`precoding<hermespy.modem.modem.BaseModem.precoding>`
attribute:

.. literalinclude:: ../scripts/examples/modem_precoding_dft.py
   :language: python
   :linenos:
   :lines: 17-42

.. autoclass:: hermespy.modem.precoding.dft.DFT

.. footbibliography::
