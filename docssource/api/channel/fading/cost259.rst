========
Cost 259
========

.. inheritance-diagram:: hermespy.channel.fading.cost259.Cost259
   :parts: 1

The Cost 259 channel model is a generic model for the simulation of mobile radio channels.
Refer to :footcite:t:`2006:MolischA` and :footcite:t:`2006:MolischB` for more information.
Parametrizations can be found in the standard :footcite:t:`3GPP:TR125943`.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_fading_cost259.py
   :language: python
   :linenos:
   :lines: 11-36

.. autoclass::  hermespy.channel.fading.cost259.Cost259

.. autoclass::  hermespy.channel.fading.cost259.Cost259Type

.. footbibliography::
