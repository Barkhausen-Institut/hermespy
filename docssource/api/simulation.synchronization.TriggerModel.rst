==============
Trigger Model
==============

.. inheritance-diagram:: hermespy.simulation.simulated_device.TriggerModel
   :parts: 1

Trigger models handle the time synchronization behaviour of devices:
Within Hermes, the :class:`SimulatedDrop<hermespy.simulation.simulation.SimulatedDrop>` models the highest level of physical layer simulation.
By default, the first sample of a drop is also considered the first sample of the contained simulation / sensing frames.
However, when multiple devices and links interfer with each other, their individual frame structures might be completely asynchronous.
This modeling can be adressed by shared trigger models, all devices sharing a trigger model will
be frame-synchronous within the simulated drop, however, different trigger models introduce unique time-offsets within the simulation drop.

The currently implemented trigger models are

.. include:: simulation.synchronization._table.rst

They can be configured by assigning the same :class:`TriggerModel<hermespy.simulation.simulated_device.TriggerModel>` instance to the :attr:`trigger_model<hermespy.simulation.simulated_device.SimulatedDevice.trigger_model>`
property of multiple simulated devices:

.. literalinclude:: ../scripts/examples/simulation_synchronization.py
   :language: python
   :lines: 10-18, 39-46

Of course, the abstract *TriggerModel* in the above snippet must be replaced by the desired implementation
from the list above.

.. autoclass:: hermespy.simulation.simulated_device.TriggerModel

.. footbibliography::
