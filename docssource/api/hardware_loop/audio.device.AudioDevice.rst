============
Audio Device
============

.. inheritance-diagram:: hermespy.hardware_loop.audio.device.AudioDevice hermespy.hardware_loop.audio.device.AudioDeviceAntennas
   :parts: 1

Hermes hardware bindings to audio devices offer the option to benchmark complex-valued
communication waveforms over affordable consumer-grade audio hardware.
The effective available bandwidth is limited to half of the audio devices sampling rate,
which is typically either :math:`44.1~\\mathrm{kHz}` or :math:`48~\\mathrm{kHz}`.

.. autoclass:: hermespy.hardware_loop.audio.device.AudioDevice

.. autoclass:: hermespy.hardware_loop.audio.device.AudioAntenna

.. autoclass:: hermespy.hardware_loop.audio.device.AudioDeviceAntennas

.. footbibliography::