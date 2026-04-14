===========
Noise Level
===========

.. inheritance-diagram:: hermespy.simulation.rf.noise.level.NoiseLevel hermespy.simulation.rf.noise.level.ThermalNoise hermespy.simulation.rf.noise.level.N0 hermespy.simulation.rf.noise.level.SNR hermespy.simulation.modem.noise.CommunicationNoiseLevel hermespy.simulation.modem.noise.EBN0 hermespy.simulation.modem.noise.ESN0 
   :parts: 1

.. autoclass:: hermespy.simulation.rf.noise.level.NoiseLevel


-------------------
General Noise Level
-------------------

.. autoclass:: hermespy.simulation.rf.noise.level.ThermalNoise

.. autoclass:: hermespy.simulation.rf.noise.level.N0

.. autoclass:: hermespy.simulation.rf.noise.level.SNR

-------------------------
Communication Noise Level
-------------------------

.. autoclass:: hermespy.simulation.modem.noise.CommunicationNoiseLevel

^^^^^
Eb/N0
^^^^^

The bit energy to noise power spectral density ratio (Eb/N0) is a fundamental parameter in digital communication systems.
It is defined as the ratio of the energy per bit to the noise power spectral density.
The Eb/N0 is a key parameter in the design of digital communication systems, and is used to determine the minimum signal-to-noise ratio required to achieve a certain bit error rate.

.. autoclass:: hermespy.simulation.modem.noise.EBN0

^^^^^
Es/N0
^^^^^

The symbol energy to noise power spectral density ratio (ES/N0) is a measure of the signal-to-noise ratio (SNR) in a communication system.
It is defined as the ratio of the energy per symbol to the noise power spectral density. The ES/N0 is a key parameter in the design of digital communication systems, and is used to determine the required signal-to-noise ratio for a given bit error rate.

.. autoclass:: hermespy.simulation.modem.noise.ESN0


.. footbibliography::