import matplotlib.pyplot as plt

from hermespy.channel import SingleTargetRadarChannel
from hermespy.jcas import OFDMRadar
from hermespy.modem import FrameResource, FrameSymbolSection, FrameElement, ElementType
from hermespy.radar import MaxDetector
from hermespy.simulation import SimulatedDevice


carrier_frequency = 24e9
oversampling_factor = 4
subcarrier_spacing = 90.909e3
num_subcarriers = 1024
prefix_ratio = 1.375 / 12.375
modulation_order = 16

device = SimulatedDevice(carrier_frequency=carrier_frequency)
resources = [FrameResource(64, prefix_ratio=prefix_ratio, elements=[FrameElement(ElementType.DATA, 15), FrameElement(ElementType.REFERENCE, 1)])]
structure = [FrameSymbolSection(11, [0])]
radar = OFDMRadar(oversampling_factor=oversampling_factor, modulation_order=modulation_order, subcarrier_spacing=subcarrier_spacing, resources=resources, structure=structure, device=device)
radar.detector = MaxDetector()

radar_channel = SingleTargetRadarChannel(.75 * radar.waveform.max_range, 1., velocity=4 * radar.velocity_resolution, attenuate=False, alpha_device=device, beta_device=device)

transmission = device.transmit()
propagation = radar_channel.propagate(transmission, device, device)
reception = device.receive(propagation)

radar.reception.cube.plot_range()
radar.reception.cube.plot_range_velocity(scale='velocity')
radar.reception.cloud.plot()

plt.show()
