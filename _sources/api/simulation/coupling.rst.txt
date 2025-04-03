===============
Mutual Coupling
===============

Mutual coupling refers to the effect of the electromagnetic field of one antenna / radio-frequency chain
on another antenna / radio-frequency chain within the same front-end.
This effect is due to the fact that the electromagnetic field of one antenna / radio-frequency chain  is not confined to the volume of the antenna / radio-frequency chain itself,
but extends to the volume of the other antenna / radio-frequency chain.
This may lead to unexpected radiation patterns, impedance mismatches, and other undesired effects.

.. note::
   Note that both mutual coupling and isolation refer to the same physical effect,
   just from different perspectives, i.e. the transmitter perspective in case of mutual
   coupling and the receiver perspective in case of isolation.
   Therefore, the two simulation steps might be merged in future releases of HermesPy.

The currently implemented mutual coupling models are

.. list-table::
   :header-rows: 1

   * - Model
     - Description

   * - :doc:`coupling.perfect`
     - Perfect coupling, i.e. no mutual coupling at all. The default model.

   * - :doc:`coupling.impedance`
     - Mutual coupling is modeled by the impedance matrix of the front-end.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>`
mutual coupling model is achived by setting the :attr:`coupling<hermespy.simulation.simulated_device.SimulatedDevice.coupling>`
property of an instance to the desired model.

.. literalinclude:: ../../scripts/examples/simulation_coupling.py
   :language: python
   :linenos:
   :lines: 6-11

Of course, the abstract *Coupling* class in this snippet has to be replaced
by the desired coupling model implementation from the above table.

.. toctree::
   :hidden:

   coupling.perfect
   coupling.impedance
   coupling.coupling

.. footbibliography::
