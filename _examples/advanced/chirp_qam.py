# -*- coding: utf-8 -*-
#
# In this example we simulate a frame of QAM modulated overlapping chirps.
# 

import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.fec import TurboCoding
from hermespy.simulation import (
    Simulation,
    RappPowerAmplifier,
    EBN0,
)
from hermespy.channel import IdealChannel
from hermespy.modem import (
    SimplexLink,
    RootRaisedCosineWaveform,
    SingleCarrierLeastSquaresChannelEstimation,
    SingleCarrierZeroForcingChannelEqualization,
    BitErrorEvaluator,
    ThroughputEvaluator,
)

# Initialize a simulation considering two devices operting in base-band
simulation = Simulation()
cf = 0.0
tx_device = simulation.new_device(carrier_frequency=cf)
rx_device = simulation.new_device(carrier_frequency=cf)

# Configure a non-ideal power Rapp power amplifier model at the transmitting device
tx_device.rf_chain.power_amplifier = RappPowerAmplifier(smoothness_factor=6.0)

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

# Configure a polar Turb coding scheme
link.encoder_manager.add_encoder(TurboCoding(40, 13, 15, 100))


# Evaluate bit error rate and throughput during simulation runtime
simulation.add_evaluator(BitErrorEvaluator(link, link, plot_surface=False))
simulation.add_evaluator(ThroughputEvaluator(link, link, plot_surface=False))

# Sweep over the receive Eb/N0 from 0 dB to 20 dB and the modulation order from 4 to 64
rx_device.noise_level = EBN0(link)
simulation.new_dimension("noise_level", dB(range(0, 21)), rx_device)
simulation.new_dimension("modulation_order", [4, 16, 64], link.waveform, title="Modulation")

# Run the simulation, plot the results
simulation.num_samples = 1000
result = simulation.run()
result.plot()
plt.show()
