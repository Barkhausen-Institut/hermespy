==================
Perfect Isolation
==================

.. inheritance-diagram:: hermespy.simulation.isolation.perfect.PerfectIsolation
   :parts: 1

The perfect isolation implementation is HermesPy's default isolation assumption,
leaking no power from transmit to receive antennas and therfore modeling perfect / ideal isolation.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>`
with a perfect isolation model is achived by setting the :attr:`isolation<hermespy.simulation.simulated_device.SimulatedDevice.isolation>`
property of an instance..

.. literalinclude:: ../scripts/examples/simulation_isolation_perfect.py
   :language: python
   :linenos:
   :lines: 5-10

.. autoclass:: hermespy.simulation.isolation.perfect.PerfectIsolation

.. footbibliography::
