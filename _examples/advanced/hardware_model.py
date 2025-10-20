# -*- coding: utf-8 -*-
# In this example we simulate the effects of a non-ideal
# Radio-Frequency chain and analog-to-digital conversion on the bit error rate performance
# of a single-carrier communication system.
# We consider a power amplifier following Rapp's model, a noisy frequency source
# and an adc with mid-riser quantization plus automatic gain control.
#
# The performance is evaluated for a signal-to-noise ratio between zero and 20 dB.

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.simulation import (
    ADC,
    Simulation,
    DAC,
    EBN0,
    Mixer,
    MixerType,
    Source,
    RappPowerAmplifier,
    PowerAmplifier,
    RFChain,
    OscillatorPhaseNoise,
    AutomaticGainControl,
    QuantizerType,
)
from hermespy.channel import IdealChannel
from hermespy.modem import (
    SimplexLink,
    RootRaisedCosineWaveform,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
    BitErrorEvaluator,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll-Barreto", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Build a customized RF chain block model
rf = RFChain()

# Common frequency source with a default phase noise model
source = rf.new_block(Source, phase_noise=OscillatorPhaseNoise())

# Transmit side
dac = rf.new_block(DAC, num_quantization_bits=16)
tx_mixer = rf.new_block(Mixer, mixer_type=MixerType.UP)
pa = rf.new_block(RappPowerAmplifier, smoothness_factor=6.0)
rf.connect(dac.o, tx_mixer.i)
rf.connect(tx_mixer.o, pa.i)
rf.connect(source.o, tx_mixer.lo)

# Receive side
adc = rf.new_block(ADC, num_quantization_bits=8, quantizer_type=QuantizerType.MID_RISER, gain=AutomaticGainControl())
rx_mixer = rf.new_block(Mixer, mixer_type=MixerType.DOWN)
lna = rf.new_block(PowerAmplifier)
rf.connect(adc.i, rx_mixer.o)
rf.connect(rx_mixer.i, lna.o)
rf.connect(source.o, rx_mixer.lo)


# Initialize a simulation considering two devices operting at 24 GHz
# Both devices use the same RF chain model
simulation = Simulation(seed=42)
device_params = {
    "rf": rf,
    "bandwidth": 100e6,
    "oversampling_factor": 4,
    "carrier_frequency": 24e9,
}
tx_device = simulation.new_device(**device_params)
rx_device = simulation.new_device(**device_params)


# Configure an ideal channel between the two devices
# This is the default setting, but we can set it explicitly
simulation.set_channel(tx_device, rx_device, IdealChannel())


# Configure a simplex link between the two devices
# The transmitted waveform is a QAM-modulated root-raised cosine chirp
link = SimplexLink(waveform=RootRaisedCosineWaveform(
    roll_off=.9,
    modulation_order=16,
    num_preamble_symbols=16,
    num_data_symbols=1024,
    pilot_rate=1e6,
    guard_interval=1e-6,
    channel_estimation=SingleCarrierLeastSquaresChannelEstimation(),
    channel_equalization=SingleCarrierZeroForcingChannelEqualization(),
))
link.connect(tx_device, rx_device)

# Evaluate the link's bit error rate during simulation runtime
simulation.add_evaluator(BitErrorEvaluator(link, link, plot_surface=False))

# Sweep over the receive Eb/N0 from 0 dB to 20 dB
rx_device.noise_level = EBN0(link)
simulation.new_dimension("noise_level", dB(range(0, 21)), rx_device)

# Sweep over the power amplifier's saturation amplitude from -10 dBV to 10 dBV
simulation.new_dimension("saturation_amplitude", dB([-10, 0, 10]), pa.block)

# Run the simulation, plot the results
simulation.num_samples = 10000
result = simulation.run()
result.plot()
plt.show()
