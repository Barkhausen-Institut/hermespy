========================
Communication Precoding
========================

Communication symbol precoders are an extension of the :doc:`precoding` module,
manipulating MIMO streams of communication symbols instead of base-band signals
during both transmission and reception.

The following types of precoders are supported:

.. toctree::
   :maxdepth: 1

   precoding.dft
   precoding.ratio_combining
   precoding.single_carrier
   precoding.space_time_block_coding

Configuring a precoder within a signal processing pipeline requires adding an instance
of said precoder to a :class:`TransmitSymbolCoding<hermespy.modem.precoding.symbol_precoding.TransmitSymbolCoding>` or :class:`ReceiveSymbolCoding<hermespy.modem.precoding.symbol_precoding.ReceiveSymbolCoding>`
pipeline, that can hold multiple precoders to be applied in sequence:

.. literalinclude:: ../../scripts/examples/modem_precoding_precoding.py
   :language: python
   :linenos:
   :lines: 14-20

Within the context of a modem,
the precodings can either be assigned to the :attr:`transmit_symbol_coding<hermespy.modem.modem.TransmittingModemBase.transmit_symbol_coding>` or  :attr:`transmit_symbol_coding<hermespy.modem.modem.ReceivingModemBase.receive_symbol_coding>`
property, or the precoders can be configured directly:

.. literalinclude:: ../../scripts/examples/modem_precoding_precoding.py
   :language: python
   :linenos:
   :lines: 25-31

Refer to :doc:`/notebooks/precoding` for instructions and examples of how to implement
custom symbol precoders.

.. toctree::
   :maxdepth: 1
   :hidden:
   :glob:

   precoding.symbol_precoding.*

.. footbibliography::
