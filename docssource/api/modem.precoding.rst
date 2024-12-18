========================
Communication Precoding
========================

Communication symbol precoders are an extension of the :doc:`precoding` module,
manipulating MIMO streams of communication symbols instead of base-band signals
during both transmission and reception.

The following types of precoders are supported:

.. toctree::
   :maxdepth: 1

   modem.precoding.dft
   modem.precoding.ratio_combining
   modem.precoding.single_carrier
   modem.precoding.space_time_block_coding

Configuring a precoder within a signal processing pipeline requires adding an instance
of said precoder to a :class:`SymbolPrecoding<hermespy.modem.precoding.symbol_precoding.SymbolPrecoding>`
pipeline, that can hold multiple precoders to be applied in sequence:

.. literalinclude:: ../scripts/examples/modem_precoding_precoding.py
   :language: python
   :linenos:
   :lines: 14-20

Within the context of a modem,
the precoding can either be assigned to the :attr:`precoding<hermespy.modem.modem.BaseModem.precoding>`
property, or the precoders can be configured directly:

.. literalinclude:: ../scripts/examples/modem_precoding_precoding.py
   :language: python
   :linenos:
   :lines: 25-31

Refer to :doc:`/notebooks/precoding` for instructions and examples of how to implement
custom symbol precoders.

.. toctree::
   :maxdepth: 1
   :hidden:
   :glob:

   modem.precoding.symbol_precoding.*

.. footbibliography::
