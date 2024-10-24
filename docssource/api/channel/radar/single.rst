=============
Single Target
=============

.. inheritance-diagram:: hermespy.channel.radar.single.SingleTargetRadarChannel hermespy.channel.radar.single.SingleTargetRadarChannelRealization
   :parts: 1

Model of a radar channel featuring a single reflecting target.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_radar_single.py
   :language: python
   :linenos:
   :lines: 11-30
    
.. autoclass:: hermespy.channel.radar.single.SingleTargetRadarChannel
   :private-members: _realize

.. autoclass:: hermespy.channel.radar.single.SingleTargetRadarChannelRealization

.. footbibliography::
