====================
Spatial Multiplexing
====================

.. inheritance-diagram:: hermespy.modem.precoding.spatial_multiplexing.SpatialMultiplexing
   :parts: 1
   :top-classes: hermespy.modem.precoding.symbol_precoding.SymbolPrecoder

Spatial multiplexing refers to processing orthogonal symbol streams in parallel.
It is essentially a stub, instructing the modem to generate, and process, multiple frames
to be transmitted in parallel. The number of frames is equal to the number of available antennas.

It configured by adding an instance to the
:class:`SymbolPrecoding<hermespy.modem.precoding.symbol_precoding.SymbolPrecoding>`
of a :class:`Modem<hermespy.modem.modem.BaseModem>` exposed by the :attr:`precoding<hermespy.modem.modem.BaseModem.precoding>`
attribute:

.. literalinclude:: ../scripts/examples/modem_precoding_sm.py
   :language: python
   :linenos:
   :lines: 18-42


.. autoclass:: hermespy.modem.precoding.spatial_multiplexing.SpatialMultiplexing

.. footbibliography::
