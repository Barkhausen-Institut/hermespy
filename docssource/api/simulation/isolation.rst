==========
Isolation
==========

Isolation, sometimes referred to as transmit-receive leakage, spillage or spillover,
refers the phenomenon of power from a frontendend's transmit chain being picked up by the receive
chain in duplex systems.
This is a common problem in RF systems and can be a limiting factor for monostatic radar performance and
full-duplex communications.

.. note::
   Note that both mutual coupling and isolation refer to the same physical effect,
   just from different perspectives, i.e. the transmitter perspective in case of mutual
   coupling and the receiver perspective in case of isolation.
   Therefore, the two simulation steps might be merged in future releases of HermesPy.

Within HermesPy, a number of isolation implementations are available, including

.. list-table::
   :header-rows: 1

   * - Isolation Model
     - Description

   * - :doc:`isolation.perfect`
     - Perfect isolation between all transmit and receive chains.

   * - :doc:`isolation.selective`
     - Frequency-seletive isolation between transmit and receive chains.

   * - :doc:`isolation.specific`
     - Scalar isolation between specific transmit and receive chains.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>`
isolation model is achived by setting the :attr:`isolation<hermespy.simulation.simulated_device.SimulatedDevice.isolation>`
property of an instance to the desired isolation model.

.. literalinclude:: ../../scripts/examples/simulation_isolation.py
   :language: python
   :linenos:
   :lines: 5-10

Of course, the abstract *Isolation* class in this snippet has to be replaced
by the desired isolation model implementation from the above table.

.. toctree::
   :hidden:
   
   isolation.isolation
   isolation.perfect
   isolation.selective
   isolation.specific

.. footbibliography::
