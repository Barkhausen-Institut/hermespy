import matplotlib.pyplot as plt

from hermespy.channel import SingleTargetRadarChannel
from hermespy.jcas import OFDMRadar
from hermespy.modem import ElementType, GridResource, PrefixType, GridElement, SymbolSection, OFDMWaveform
from hermespy.radar import MaxDetector
from hermespy.simulation import SimulatedDevice

# Configure a OFDM radar
device = SimulatedDevice(carrier_frequency=24e9)
radar = OFDMRadar(OFDMWaveform(
    grid_resources=[
        GridResource(16, PrefixType.CYCLIC, .1, [GridElement(ElementType.DATA, 7), GridElement(ElementType.REFERENCE, 1)]),
        GridResource(128, PrefixType.CYCLIC, .1, [GridElement(ElementType.DATA, 1)]),
    ],
    grid_structure=[
        SymbolSection(64, [0, 1])
    ],
    num_subcarriers=1024,
    subcarrier_spacing=90.909e3,
    oversampling_factor=4,
))
radar.detector = MaxDetector()
device.add_dsp(radar)

# Generate a single target
radar_channel = SingleTargetRadarChannel(.75 * radar.max_range, 1., velocity=10, attenuate=False)
transmission = device.transmit()
propagation = radar_channel.propagate(transmission, device, device)
reception = device.receive(propagation)

# Visualize radar image
reception.operator_receptions[0].cube.plot_range()
reception.operator_receptions[0].cube.plot_range_velocity(scale='velocity')
reception.operator_receptions[0].cloud.visualize()

plt.show()
