# -*- coding: utf-8 -*-
# In this example we evaluate the direct current power consumption of a
# radio-frequency chain and the resulting efficiency of its power amplifier.
#
# Both the amplifier's distortion characteristic and its supply draw come from
# the same measurement of a class-C part: the gain compresses from 13.4 down to
# 1.9 as the amplifier is driven harder, while the supply draw rises from a
# quiescent 87 mW to 218 mW.
#
# Sweeping the level at which the digital-to-analog converter drives the chain
# exposes the trade-off. Driven softly, the amplifier still draws its quiescent
# power while emitting almost nothing, so the efficiency is negligible. Driven
# hard, the efficiency approaches the amplifier's ceiling, but the compression
# distorts the constellation and the bit error rate collapses. The useful
# operating point is the last one before that happens.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import dB
from hermespy.simulation import (
    ADC,
    AutomaticGainControl,
    ConstantDCPowerModel,
    CustomPowerAmplifier,
    DAC,
    Mixer,
    MixerType,
    PowerAmplifier,
    PowerConsumptionEvaluator,
    QuantizerType,
    RFChain,
    SampledDCPowerModel,
    Simulation,
    Source,
)
from hermespy.channel import IdealChannel
from hermespy.modem import (
    BitErrorEvaluator,
    RootRaisedCosineWaveform,
    SimplexLink,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
)

__author__ = "Emre Can Ataş"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Emre Can Ataş", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Load the measured amplifier characteristics and convert them to SI units
DATASHEET = Path(__file__).parent.parent / "resources" / "class_c_amplifier.csv"
measurements = np.loadtxt(DATASHEET, delimiter=",", comments="#")

input_powers = 10 ** (measurements[:, 0] / 10) / 1000  # dBm to Watt
output_powers = 10 ** (measurements[:, 1] / 10) / 1000  # dBm to Watt
consumed_powers = measurements[:, 2] / 1000  # milliwatt to Watt

# The distortion characteristic, as a voltage gain per input amplitude
amplifier_amplitudes = np.sqrt(input_powers)
amplifier_gains = np.sqrt(output_powers / input_powers)
amplifier_phases = np.zeros_like(amplifier_gains)  # no phase distortion measured

# The supply draw at each of the same operating points
amplifier_consumption = SampledDCPowerModel(input_powers, consumed_powers)


# Build a radio-frequency chain whose active blocks declare what they consume
rf = RFChain()

# Common frequency source, drawing a fixed bias current
source = rf.new_block(Source, dc_power_model=ConstantDCPowerModel(0.02))

# Transmit side
dac = rf.new_block(DAC, num_quantization_bits=16)
tx_mixer = rf.new_block(Mixer, mixer_type=MixerType.UP)
pa = rf.new_block(
    CustomPowerAmplifier,
    input=amplifier_amplitudes,
    gains=amplifier_gains,
    phases=amplifier_phases,
    dc_power_model=amplifier_consumption,
)
rf.connect(dac.o, tx_mixer.i)
rf.connect(tx_mixer.o, pa.i)
rf.connect(source.o, tx_mixer.lo)

# Receive side
adc = rf.new_block(
    ADC,
    num_quantization_bits=8,
    quantizer_type=QuantizerType.MID_RISER,
    gain=AutomaticGainControl(),
)
rx_mixer = rf.new_block(Mixer, mixer_type=MixerType.DOWN)
lna = rf.new_block(PowerAmplifier, dc_power_model=ConstantDCPowerModel(0.01))
rf.connect(adc.i, rx_mixer.o)
rf.connect(rx_mixer.i, lna.o)
rf.connect(source.o, rx_mixer.lo)


# Initialize a simulation of two devices operating at 24 GHz over an ideal channel,
# so that every bit error traces back to the amplifier
simulation = Simulation(seed=42)
device_params = {
    "rf": rf,
    "bandwidth": 100e6,
    "oversampling_factor": 4,
    "carrier_frequency": 24e9,
}
tx_device = simulation.new_device(**device_params)
rx_device = simulation.new_device(**device_params)
simulation.set_channel(tx_device, rx_device, IdealChannel())


# Transmit a QAM-modulated root-raised cosine waveform between the two devices.
# Sixteen constellation points carry information in the amplitude as well as the
# phase, so compressing the amplifier actually costs bits.
link = SimplexLink(
    waveform=RootRaisedCosineWaveform(
        roll_off=0.9,
        modulation_order=16,
        num_preamble_symbols=16,
        num_data_symbols=256,
        pilot_rate=1e6,
        guard_interval=1e-6,
        channel_estimation=SingleCarrierLeastSquaresChannelEstimation(),
        channel_equalization=SingleCarrierZeroForcingChannelEqualization(),
    )
)
link.connect(tx_device, rx_device)


# Evaluate the link quality alongside the amplifier's efficiency
simulation.add_evaluator(BitErrorEvaluator(link, link, plot_surface=False, plot_scale="linear"))
simulation.add_evaluator(PowerConsumptionEvaluator(pa.block, plot_surface=False))


# Sweep the level at which the converter drives the chain, spanning the input
# power range the measurement covers
simulation.new_dimension(
    "max_output_power", dB([-75, -70, -65, -60, -55, -50, -45, -40]), dac.block
)


# Run the simulation, plot the results
simulation.num_samples = 20
result = simulation.run()
result.plot()
plt.show()
