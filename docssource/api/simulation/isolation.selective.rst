===================
Selective Leakage
===================

.. inheritance-diagram:: hermespy.simulation.isolation.selective.SelectiveLeakage
   :parts: 1

The selective leakage implementation allows for the definition of frequency-domain
filter characteristics for each transmit-receive antenna pair within a radio-frequency frontend.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>`
with a selective isolation model is achived by setting the :attr:`isolation<hermespy.simulation.simulated_device.SimulatedDevice.isolation>`
property of an instance.

.. literalinclude:: ../../scripts/examples/simulation_isolation_selective.py
   :language: python
   :linenos:
   :lines: 8-17

.. autoclass:: hermespy.simulation.isolation.selective.SelectiveLeakage

.. footbibliography::
