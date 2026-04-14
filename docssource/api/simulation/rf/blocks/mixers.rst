======
Mixers
======

.. inheritance-diagram:: hermespy.simulation.rf.blocks.mixers.Mixer hermespy.simulation.rf.blocks.mixers.IdealMixer
   :parts: 1

Radio-Frequency (RF) mixers are components combining two input signals by multipling them,
resulting in sum and difference frequency components at the output.
In the process, they commonly introduce non-linear distortion, phase shifts, thermal noise and
leakage of the local oscillator signal to the output.

This module provides various numerical models of RF mixers.

.. autoclass:: hermespy.simulation.rf.blocks.mixers.Mixer

.. autoclass:: hermespy.simulation.rf.blocks.mixers.IdealMixer

.. autoclass:: hermespy.simulation.rf.blocks.mixers.MixerType

.. autoclass:: hermespy.simulation.rf.blocks.mixers.IdealMixerRealization

.. autoclass:: hermespy.simulation.rf.blocks.mixers.MixerRealization
