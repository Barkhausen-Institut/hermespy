=================
Synchronization
=================

.. inheritance-diagram:: hermespy.modem.waveform.Synchronization
   :parts: 1

Synchronization in Hermes refers to the process of partitioning streams of received base-band samples
into communication frames.
Synchronization can be interpreted as an optimization problem

.. math::

   \underset{\tau}{\text{maximize}} ~ \lVert \mathbf{H}(t, \tau) \rVert_\mathrm{F} \quad \text{for} \quad T_\mathrm{min} \leq t \leq T_\mathrm{max}

estimating the primary delay component :math:`\tau` of the channel model :math:`\mathbf{H}(t, \tau)` within specific interval between :math:`T_\mathrm{min}` and :math:`T_\mathrm{max}`.
The estimated delay is equivalent to a perceived timing offset between the transmitter's and receiver's respective clocks.
Most statistical channel models for link-level simulations ignore the minimum free-space propagation delays of waveforms traveling from one device to another and instead model only the delay spread of multipath components.
As a result, the first signal sample after channel propagation contains the first sample of the propagated signal's line of sight component, or, in non line of sight cases, the first sample of the shortest path propagation.
Therefore, link-level simulations tend to ignore synchronization and only focus on processing the multipath components.
When considering spatial channel models with realistic propagation delays or, more importantly, transmitting waveforms over real hardware, estimating the introduced propagation delays becomes vital for error-free information transmission.

.. autoclass:: hermespy.modem.waveform.Synchronization

.. toctree::
   :hidden:

   modem.waveform.correlation_synchronization

.. footbibliography::
