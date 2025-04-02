======================
Sample Offset Trigger
======================

.. inheritance-diagram:: hermespy.simulation.simulated_device.SampleOffsetTrigger
   :parts: 1

The :class:`SampleOffsetTrigger<hermespy.simulation.simulated_device.SampleOffsetTrigger>`
introduces an offset of a fixed number of samples to the start of the :class:`SimulatedDrop<hermespy.simulation.drop.SimulatedDrop>`.

It can be configured by assigning the same :class:`SampleOffsetTrigger<hermespy.simulation.simulated_device.SampleOffsetTrigger>` instance to the :attr:`trigger_model<hermespy.simulation.simulated_device.SimulatedDevice.trigger_model>`
property of multiple simulated devices:

.. literalinclude:: ../../scripts/examples/simulation_synchronization_SampleOffsetTrigger.py
   :language: python
   :lines: 10-18, 39-46

.. autoclass:: hermespy.simulation.simulated_device.SampleOffsetTrigger

.. footbibliography::
