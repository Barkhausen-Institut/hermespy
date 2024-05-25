==============
Indoor Factory
==============

.. inheritance-diagram:: hermespy.channel.cdl.indoor_factory.IndoorFactory hermespy.channel.cdl.indoor_factory.IndoorFactoryRealization
   :parts: 1

Channel implementation of an indoor factory scenario.
Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_cdl_indoor_factory.py
   :language: python
   :linenos:
   :lines: 12-40

.. autoclass:: hermespy.channel.cdl.indoor_factory.IndoorFactory

.. autoclass:: hermespy.channel.cdl.indoor_factory.IndoorFactoryRealization

.. autoclass:: hermespy.channel.cdl.indoor_factory.FactoryType

.. footbibliography::
