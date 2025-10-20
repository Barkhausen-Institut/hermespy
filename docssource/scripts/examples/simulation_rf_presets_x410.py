# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.core import AntennaMode, Transformation
from hermespy.channel import UrbanMacrocells, O2IState
from hermespy.simulation.rf.presets.ettus.x410 import X410
from hermespy.simulation import StaticTrajectory, SimulationScenario, SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.modem import SimplexLink, OFDMWaveform, GridResource, GridElement, ElementType, SymbolSection, OrthogonalLeastSquaresChannelEstimation, OrthogonalZeroForcingChannelEqualization


carrier_frequency = 6e9

scenario = SimulationScenario(seed=42)
devices = [scenario.new_device(
    carrier_frequency=carrier_frequency,
    bandwidth=1024*15e3,
) for _ in range(2)]

for device in scenario.devices:
    device.rf = X410(
        carrier_frequency=carrier_frequency,
        tx_gain=40.0,
        rx_gain=40.0,
    )
    device.antennas = SimulatedUniformArray(SimulatedIdealAntenna(AntennaMode.DUPLEX), .1, [1, 1, 1])

scenario.devices[0].trajectory = StaticTrajectory(Transformation.From_Translation([0, 0, 6]))
scenario.devices[1].trajectory = StaticTrajectory(Transformation.From_Translation([50, 50, 6]))

channel = UrbanMacrocells(expected_state=O2IState.LOS)
scenario.set_channel(scenario.devices[0], scenario.devices[1], channel)


link = SimplexLink(
    selected_transmit_ports=[0],
    selected_receive_ports=[0],
    waveform=OFDMWaveform(
        [
            GridResource(512, prefix_ratio=0.375, elements=[
                GridElement(ElementType.DATA, 1),
                GridElement(ElementType.REFERENCE, 1),
            ]),
            GridResource(512, prefix_ratio=0.375, elements=[
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 1),
            ]),
        ],
        [SymbolSection(1, [0, 1])],
        num_subcarriers=1024,
        modulation_order=4,
        channel_estimation=OrthogonalLeastSquaresChannelEstimation(),
        channel_equalization=OrthogonalZeroForcingChannelEqualization(),
    )
)
link.connect(scenario.devices[0], scenario.devices[1])

drop = scenario.drop()
drop.device_transmissions[0].mixed_signal.plot(title="Tx RF Signal")
drop.device_receptions[1].operator_inputs[0].plot(title="Rx BB Signal")
drop.device_receptions[1].operator_receptions[0].equalized_symbols.plot_constellation()
plt.show()
