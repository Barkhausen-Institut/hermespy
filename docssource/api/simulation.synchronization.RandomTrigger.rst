==============
Random Trigger
==============

.. inheritance-diagram:: hermespy.simulation.simulated_device.RandomTrigger
   :parts: 1

The random trigger model introduces a synchronization offset that is uniformly
distributed over the interval :math:`[0, T_{\mathrm{Frame}}]` where :math:`T_{\mathrm{Frame}}`
denotes the maximal frame duration of all devices controlled by the same trigger instance.

It can be configured by assigning the same :class:`RandomTrigger<hermespy.simulation.simulated_device.RandomTrigger>` instance to the :attr:`trigger_model<hermespy.simulation.simulated_device.SimulatedDevice.trigger_model>`
property of multiple simulated devices:

.. literalinclude:: ../scripts/examples/simulation_synchronization_RandomTrigger.py
   :language: python
   :lines: 10-18, 39-46

.. autoclass:: hermespy.simulation.simulated_device.RandomTrigger

.. footbibliography::
