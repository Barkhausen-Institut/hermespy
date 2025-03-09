# -*- coding: utf-8 -*-
# In this example we simulate the effects of a non-ideal
# Radio-Frequency chain and analog-to-digital conversion on the bit error rate performance
# of a single-carrier communication system.
# We consider I/Q imbalance, a power amplifier following Rapp's model and an adc with
# mid-riser quantization and automatic gain control.
# 
# The performance is evaluated for a signal-to-noise ratio between zero and 20 dB.

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.simulation import (
    Simulation,
    EBN0,
    RappPowerAmplifier,
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


# Initialize a simulation considering two devices operting in base-band
simulation = Simulation()
cf = 0.0
tx_device = simulation.new_device(carrier_frequency=cf)
rx_device = simulation.new_device(carrier_frequency=cf)

# Configure a non-ideal power Rapp power amplifier model at the transmitting device
tx_device.rf_chain.power_amplifier = RappPowerAmplifier(smoothness_factor=6.0)

# Configure phase noisy oscillators, quantization, gain control and amplitude imbalances
tx_device.rf_chain.oscillator_phase_noise = OscillatorPhaseNoise()
tx_device.rf_chain.amplitude_imbalance = 1e-3
tx_device.rf_chain.phase_offset = 1e-2
rx_device.rf_chain.osillator_phase_noise = OscillatorPhaseNoise()
rx_device.rf_chain.amplitude_imbalance = 1e-3
rx_device.rf_chain.phase_offset = 1e-2
rx_device.rf_chain.adc.gain = AutomaticGainControl()
rx_device.rf_chain.adc.quantizer_type = QuantizerType.MID_RISER

# Configure an ideal channel between the two devices
# This is the default setting, but we can set it explicitly
simulation.set_channel(tx_device, rx_device, IdealChannel())

# Configure a simplex link between the two devices
# The transmitted waveform is a QAM-modulated root-raised cosine chirp
link = SimplexLink(waveform=RootRaisedCosineWaveform(
    roll_off=.9,
    modulation_order=16,
    symbol_rate=100e6,
    oversampling_factor=4,
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

# Run the simulation, plot the results
simulation.num_samples = 1000
result = simulation.run()
result.plot()
plt.show()
