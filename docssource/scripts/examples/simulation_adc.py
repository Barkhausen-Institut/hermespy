# -*- coding: utf-8 -*-

from hermespy.simulation import Simulation, AnalogDigitalConverter, AutomaticGainControl, QuantizerType


# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Configure a custom analog-digital conversion model
adc = AnalogDigitalConverter()
device.rf_chain.adc = adc

# Further configure the device's ADC
adc.num_quantization_bits = 16
adc.quantizer_type = QuantizerType.MID_RISER
adc.gain = AutomaticGainControl()
