===============
Normal Trigger
===============

.. inheritance-diagram:: hermespy.simulation.simulated_device.NormalTrigger
   :parts: 1

The :class:`NormalTrigger<hermespy.simulation.simulated_device.NormalTrigger>`
introduces a normally distributed random synchronization delay between the triggering event and the actual transmission or reception of a frame by a simulated device.

It can be configured by assigning the same :class:`NormalTrigger<hermespy.simulation.simulated_device.NormalTrigger>` instance to the :attr:`trigger_model<hermespy.simulation.simulated_device.SimulatedDevice.trigger_model>`
property of multiple simulated devices:

.. literalinclude:: ../../scripts/examples/simulation_synchronization_NormalTrigger.py
   :language: python
   :lines: 10-18, 37-44

.. autoclass:: hermespy.simulation.simulated_device.NormalTrigger

.. footbibliography::
