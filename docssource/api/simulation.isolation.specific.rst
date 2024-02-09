==================
Specific Isolation
==================

.. inheritance-diagram:: hermespy.simulation.isolation.specific.SpecificIsolation
   :parts: 1

The specific isolation implementation is a simplified isolation assumption assuming a scalar
relationship between transmitted waveforms and waveforms leaking into the receiving radio-frequency chains.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>`
with a specific isolation model is achived by setting the :attr:`isolation<hermespy.simulation.simulated_device.SimulatedDevice.isolation>`
property of an instance.

.. literalinclude:: ../scripts/examples/simulation_isolation_selective.py
   :language: python
   :linenos:
   :lines: 8-17


.. autoclass:: hermespy.simulation.isolation.specific.SpecificIsolation

.. footbibliography::
