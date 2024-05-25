================
Rural Macrocells
================

.. inheritance-diagram:: hermespy.channel.cdl.rural_macrocells.RuralMacrocells hermespy.channel.cdl.rural_macrocells.RuralMacrocellsRealization
   :parts: 1

Implementation of a rural macrocell communication channel model.
Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_cdl_rural_macrocells.py
   :language: python
   :linenos:
   :lines: 12-40
    

.. autoclass:: hermespy.channel.cdl.rural_macrocells.RuralMacrocells

.. autoclass:: hermespy.channel.cdl.rural_macrocells.RuralMacrocellsRealization

.. footbibliography::
