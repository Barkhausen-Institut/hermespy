======
5G TDL
======

.. inheritance-diagram:: hermespy.channel.fading.tdl.TDL
   :parts: 1


Implementation of the 3GPP standard parameterizations as stated in ETSI TR 38.900 :footcite:p:`3GPP:TR38901`.
Five scenario types A-E are defined, differing in the number of considered paths and the path's
respective delay and power.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_MultipathFading5GTDL.py
   :language: python
   :linenos:
   :lines: 11-37

.. autoclass:: hermespy.channel.fading.tdl.TDL

.. autoclass:: hermespy.channel.fading.tdl.TDLType

.. footbibliography::
