===================
Time Offset Trigger
===================

.. inheritance-diagram:: hermespy.simulation.simulated_device.TimeOffsetTrigger
   :parts: 1

The :class:`TimeOffsetTrigger<hermespy.simulation.simulated_device.TimeOffsetTrigger>`
introduces an offset of a fixed time duration to the start of the :class:`SimulatedDrop<hermespy.simulation.drop.SimulatedDrop>`.

It can be configured by assigning the same :class:`TimeOffsetTrigger<hermespy.simulation.simulated_device.TimeOffsetTrigger>` instance to the :attr:`trigger_model<hermespy.simulation.simulated_device.SimulatedDevice.trigger_model>`
property of multiple simulated devices:

.. literalinclude:: ../../scripts/examples/simulation_synchronization_TimeOffsetTrigger.py
   :language: python
   :lines: 10-18, 39-46

.. autoclass:: hermespy.simulation.simulated_device.TimeOffsetTrigger

.. footbibliography::
