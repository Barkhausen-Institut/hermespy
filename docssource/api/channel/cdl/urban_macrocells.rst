================
Urban Macrocells
================

.. inheritance-diagram:: hermespy.channel.cdl.urban_macrocells.UrbanMacrocells hermespy.channel.cdl.urban_macrocells.UrbanMacrocellsRealization
   :parts: 1

Implementation of an urban macrocell channel model.
Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_cdl_urban_macrocells.py
   :language: python
   :linenos:
   :lines: 12-40

.. autoclass:: hermespy.channel.cdl.urban_macrocells.UrbanMacrocells

.. autoclass:: hermespy.channel.cdl.urban_macrocells.UrbanMacrocellsRealization

.. footbibliography::
