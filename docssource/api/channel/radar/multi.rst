=============
Multi Target
=============

.. inheritance-diagram:: hermespy.channel.radar.multi.MultiTargetRadarChannel hermespy.channel.radar.multi.MultiTargetRadarChannelRealization
   :parts: 1

Model of a spatial radar channel featuring multiple reflecting targets.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_radar_multi.py
   :language: python
   :linenos:
   :lines: 11-43


.. autoclass:: hermespy.channel.radar.multi.MultiTargetRadarChannel
   :private-members: _realize

.. autoclass:: hermespy.channel.radar.multi.MultiTargetRadarChannelRealization

.. autoclass:: hermespy.channel.radar.multi.RadarTarget

.. autoclass:: hermespy.channel.radar.multi.VirtualRadarTarget

.. autoclass:: hermespy.channel.radar.multi.PhysicalRadarTarget

.. autoclass:: hermespy.channel.radar.multi.RadarCrossSectionModel

.. autoclass:: hermespy.channel.radar.multi.FixedCrossSection

.. footbibliography::
