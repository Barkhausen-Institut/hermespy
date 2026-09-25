# -*- coding: utf-8 -*-

import numpy as np

from hermespy.channel import IdealChannel
from hermespy.modem import RootRaisedCosineWaveform, SimplexLink
from hermespy.simulation import (
    ADC,
    AutomaticGainControl,
    ConstantDCPowerModel,
    DAC,
    DCPowerConsumptionEvaluator,
    Mixer,
    MixerType,
    PowerAmplifier,
    PowerConsumptionEvaluator,
    RFChain,
    SampledDCPowerModel,
    Simulation,
    Source,
    WaldenADCPowerModel,
)


# Build a radio-frequency chain whose active blocks declare what they draw
# from the supply. Passive blocks draw nothing and need no model.
rf = RFChain()

# The local oscillator draws a fixed bias current, no matter what the chain carries.
source = rf.new_block(Source, dc_power_model=ConstantDCPowerModel(20e-3))

# The amplifier's draw follows the drive level. A datasheet lists the supply
# power at a handful of operating points, interpolated in between. The values
# below are illustrative round numbers for a class-C part.
amplifier_draw = SampledDCPowerModel(
    np.array([1e-6, 1e-5, 1e-4, 1e-3]),        # input power in Watt, strictly ascending
    np.array([80e-3, 100e-3, 140e-3, 220e-3]),  # consumed direct current power in Watt
)

# Transmit side
dac = rf.new_block(DAC, num_quantization_bits=16)
tx_mixer = rf.new_block(Mixer, mixer_type=MixerType.UP)
pa = rf.new_block(PowerAmplifier, dc_power_model=amplifier_draw)
rf.connect(dac.o, tx_mixer.i)
rf.connect(tx_mixer.o, pa.i)
rf.connect(source.o, tx_mixer.lo)

# Receive side
lna = rf.new_block(PowerAmplifier, dc_power_model=ConstantDCPowerModel(10e-3))
rx_mixer = rf.new_block(Mixer, mixer_type=MixerType.DOWN)
# The converter's draw follows from its resolution and bandwidth rather than
# from a datasheet, so no measurement is needed to model it.
adc = rf.new_block(
    ADC,
    num_quantization_bits=8,
    gain=AutomaticGainControl(),
    dc_power_model=WaldenADCPowerModel(8, 100e6),
)
rf.connect(rx_mixer.i, lna.o)
rf.connect(adc.i, rx_mixer.o)
rf.connect(source.o, rx_mixer.lo)

# Two devices sharing the chain layout, linked over an ideal channel
simulation = Simulation(seed=42)
device_params = {"rf": rf, "bandwidth": 100e6, "carrier_frequency": 24e9}
tx_device = simulation.new_device(**device_params)
rx_device = simulation.new_device(**device_params)
simulation.set_channel(tx_device, rx_device, IdealChannel())

link = SimplexLink(waveform=RootRaisedCosineWaveform(
    modulation_order=16, num_preamble_symbols=16, num_data_symbols=64
))
link.connect(tx_device, rx_device)

# Evaluate the amplifier's total efficiency, P_out / (P_in + P_DC)
simulation.add_evaluator(PowerConsumptionEvaluator(pa.block, plot_surface=False))

# For the converter, report the drawn supply power instead. Quantization leaves
# the signal power almost unchanged, so its efficiency would sit near one
# whatever the resolution.
simulation.add_evaluator(DCPowerConsumptionEvaluator(adc.block, plot_surface=False))

simulation.num_samples = 10
simulation.run()

# The converter's draw is four orders of magnitude below the amplifier's, so it
# rounds to zero in the result table. Read it from the model instead.
print(f"Converter supply power: {adc.block.dc_power_model.power * 1e6:.1f} microwatt")
