======
State
======

.. inheritance-diagram:: hermespy.core.state.State hermespy.core.state.DeviceState hermespy.core.state.TransmitState hermespy.core.state.ReceiveState
   :parts: 1

State objects represent the immutable properties of a device at a specific point in time.
They are being passed in between the different stages of processing and simulation pipelines instead of references to the actual device objects.

.. autoclass:: hermespy.core.state.State

.. autoclass:: hermespy.core.state.DeviceState

.. autoclass:: hermespy.core.state.TransmitState

.. autoclass:: hermespy.core.state.ReceiveState

.. autoclass:: hermespy.core.state.DST

.. footbibliography::