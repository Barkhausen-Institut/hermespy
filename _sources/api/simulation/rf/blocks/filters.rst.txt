=======
Filters
=======

.. inheritance-diagram:: hermespy.simulation.rf.blocks.filters.Filter hermespy.simulation.rf.blocks.filters.HPF
   :parts: 1

Radio-Frequency (RF) filters selectively pass spectral signal components of desired frequency ranges while attenuating undesired frequencies.
They are commonly used to remove out-of-band noise, suppress interference, and shape signal spectra in communication and radar systems.

This module provides various numerical models of filters commonly applied in front-ends.

.. autoclass:: hermespy.simulation.rf.blocks.filters.HPF

.. autoclass:: hermespy.simulation.rf.blocks.filters.Filter

.. autoclass:: hermespy.simulation.rf.blocks.filters.FilterRealization
