# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import dB, AntennaMode, Transformation
from hermespy.channel import UrbanMacrocells, O2IState
from hermespy.simulation.rf.presets.ettus.x410 import X410
from hermespy.simulation import StaticTrajectory, SimulationScenario, SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.modem import nr_bandwidth, NRSlot, SimplexLink, OrthogonalLeastSquaresChannelEstimation, OrthogonalZeroForcingChannelEqualization


cf = 3.7e9

scenario = SimulationScenario(seed=42)
for _ in range(2):

    # Add a new device
    device = scenario.new_device(
            carrier_frequency=cf,
            bandwidth=nr_bandwidth(numerology=0),
            oversampling_factor=3,
            rf=X410(
                carrier_frequency=cf,
                tx_gain=dB(35),
                rx_gain=dB(10),
            ),
            antennas=SimulatedUniformArray(
                SimulatedIdealAntenna(AntennaMode.DUPLEX),
                .1,
                [1, 1, 1],
            ),
        )

# 3GPP Urban Macrocell channel model
channel = UrbanMacrocells(expected_state=O2IState.LOS)
scenario.set_channel(scenario.devices[0], scenario.devices[1], channel)
scenario.devices[0].trajectory = StaticTrajectory(Transformation.From_Translation([0, 0, 6]))
scenario.devices[1].trajectory = StaticTrajectory(Transformation.From_Translation([50, 50, 1.5]))

# Downlink with a single NR slot
slot = NRSlot()
slot.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
slot.channel_equalization = OrthogonalZeroForcingChannelEqualization()

link = SimplexLink(
    selected_transmit_ports=[0],
    selected_receive_ports=[0],
    waveform=slot,
)
link.connect(scenario.devices[0], scenario.devices[1])

# Generate a single drop and plot the transmitted and received signals
drop = scenario.drop()
drop.device_transmissions[0].mixed_signal.plot(title="Tx RF Signal")
drop.device_receptions[1].operator_inputs[0].plot(title="Rx BB Signal")
drop.device_receptions[1].operator_receptions[0].equalized_symbols.plot_constellation()
plt.show()
