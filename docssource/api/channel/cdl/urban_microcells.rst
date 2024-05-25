================
Urban Microcells
================

.. inheritance-diagram:: hermespy.channel.cdl.urban_microcells.UrbanMicrocells hermespy.channel.cdl.urban_microcells.UrbanMicrocellsRealization
   :parts: 1

Model of an urban street canyon.
Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_cdl_street_canyon.py
   :language: python
   :linenos:
   :lines: 12-40
    
.. autoclass:: hermespy.channel.cdl.urban_microcells.UrbanMicrocells

.. autoclass:: hermespy.channel.cdl.urban_microcells.UrbanMicrocellsRealization

.. footbibliography::
