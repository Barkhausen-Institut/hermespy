=========================
Analog-Digital Conversion
=========================

.. inheritance-diagram:: hermespy.simulation.rf.blocks.ad.ConverterBase hermespy.simulation.rf.blocks.ad.ADC hermespy.simulation.rf.blocks.ad.DAC
   :parts: 1

Analog-digital conversion refers to the process of converting continous electromagnetic signals into discrete digital signals that can be processed by digital systems.
Vice versa, digital-analog conversion is the process of converting discrete digital signals back into continuous analog signals.
In the process of conversion, various impairments such as quantizzation noise, thermal noise, jitter and non-linear distortion may be introduced.

This module provides blocks modeling the respective conversion harware components.

.. autoclass:: hermespy.simulation.rf.blocks.ad.ADC

.. autoclass:: hermespy.simulation.rf.blocks.ad.DAC

.. autoclass:: hermespy.simulation.rf.blocks.ad.ConverterBase

.. autoclass:: hermespy.simulation.rf.blocks.ad.GainControlBase

.. autoclass:: hermespy.simulation.rf.blocks.ad.Gain

.. autoclass:: hermespy.simulation.rf.blocks.ad.AutomaticGainControl

.. autoclass:: hermespy.simulation.rf.blocks.ad.GainType

.. autoclass:: hermespy.simulation.rf.blocks.ad.GainControlType

.. autoclass:: hermespy.simulation.rf.blocks.ad.QuantizerType
