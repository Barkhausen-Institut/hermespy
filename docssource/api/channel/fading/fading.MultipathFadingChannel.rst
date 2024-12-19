========================
Multipath Fading Channel
========================

.. inheritance-diagram:: hermespy.channel.fading.fading.MultipathFadingChannel
   :parts: 1


Allows for the direct configuration of the Multipath Fading Channel's parameters

.. math::

   \mathbf{g} &= \left[ g_{1}, g_{2}, \,\dotsc,\, g_{L}  \right]^\mathsf{T} \in \mathbb{C}^{L} \\
   \mathbf{k} &= \left[ K_{1}, K_{2}, \,\dotsc,\, K_{L}  \right]^\mathsf{T} \in \mathbb{R}^{L} \\
   \mathbf{\tau} &= \left[ \tau_{1}, \tau_{2}, \,\dotsc,\, \tau_{L}  \right]^\mathsf{T} \in \mathbb{R}^{L} \\

directly.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_fading_fading.py
   :language: python
   :linenos:
   :lines: 12-40

.. autoclass:: hermespy.channel.fading.fading.MultipathFadingChannel

.. footbibliography::
