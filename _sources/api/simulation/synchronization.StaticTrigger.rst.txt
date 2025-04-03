==============
Static Trigger
==============

.. inheritance-diagram:: hermespy.simulation.simulated_device.StaticTrigger
   :parts: 1

The :class:`StaticTrigger<hermespy.simulation.simulated_device.StaticTrigger>` will synchronize
all controlled simulated devices to be triggered exactly at the start of the :class:`SimulatedDrop<hermespy.simulation.drop.SimulatedDrop>`.

It can be configured by assigning the same :class:`StaticTrigger<hermespy.simulation.simulated_device.TimeOffsetTrigger>` instance to the :attr:`trigger_model<hermespy.simulation.simulated_device.SimulatedDevice.trigger_model>`
property of multiple simulated devices:

.. literalinclude:: ../../scripts/examples/simulation_synchronization_StaticTrigger.py
   :language: python
   :lines: 10-18, 39-46

.. autoclass:: hermespy.simulation.simulated_device.StaticTrigger

.. footbibliography::
