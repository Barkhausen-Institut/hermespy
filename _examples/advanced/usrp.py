# -*- coding: utf-8 -*-
#
# This example configures a set of Universal Software Defined Radios (USRPs)
# to establish a SISO link exchanging 10 frames of OFDM data.
#
# Operating USRP's requires additional dependencies to be installed.
# Make sure you ran pip install "hermespy[uhd]"
# See https://hermespy.org/installation.html for further details.

from rich.console import Console

from hermespy.hardware_loop import (
    HardwareLoop,
    UsrpDevice,
    UsrpSystem,
    PhysicalDeviceDummy,
    PhysicalScenarioDummy,
    ArtifactPlot,
    DeviceTransmissionPlot,
    DeviceReceptionPlot,
    ReceivedConstellationPlot,
)
from hermespy.modem import (
    BitErrorEvaluator,
    DuplexModem,
    OFDMWaveform,
    SchmidlCoxPilotSection,
    SchmidlCoxSynchronization,
    OrthogonalLeastSquaresChannelEstimation,
    OrthogonalZeroForcingChannelEqualization,
    GridElement,
    PrefixType,
    GridResource,
    ElementType,
    SymbolSection,
    GuardSection,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Open a new rich console and query the user for the USRP's IP
console = Console()
ip = console.input("Enter the IP address of the USRP (leave empty for demonstration mode): ").strip()

carrier_frequency = 1e9

# If the user did not provide an IP, we use a dummy scenario
if ip == "":
    loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](PhysicalScenarioDummy())
    device = loop.new_device(carrier_frequency=carrier_frequency)

# If an IP was provided, we attempt to connect to the USRP
else:
    loop = HardwareLoop[UsrpSystem, UsrpDevice](UsrpSystem())
    device = loop.new_device(
        ip=ip,
        port=5555,
        carrier_frequency=carrier_frequency,
        tx_gain=20.0,
        rx_gain=20.0,
    )

# Configure an OFDM modem
modem = DuplexModem(waveform=OFDMWaveform(
    modulation_order=4,
    dc_suppression=False,
    num_subcarriers=4096,
    channel_estimation=OrthogonalLeastSquaresChannelEstimation(),
    channel_equalization=OrthogonalZeroForcingChannelEqualization(),
    synchronization=SchmidlCoxSynchronization(),
    grid_resources=[
        GridResource(100, PrefixType.CYCLIC, 0.0703125, [
            GridElement(ElementType.REFERENCE, 1),
            GridElement(ElementType.DATA, 20),
        ]),
        GridResource(100, PrefixType.CYCLIC, 0.078125, [
            GridElement(ElementType.REFERENCE, 1),
            GridElement(ElementType.DATA, 20),
        ]),
    ],
    grid_structure=[
        SchmidlCoxPilotSection(),
        SymbolSection(1, [1, 0, 0, 0]),
        GuardSection(35.677083e-6, 3),
        SymbolSection(1, [1, 0, 0, 0]),
        GuardSection(35.677083e-6, 3),
    ],
))
device.add_dsp(modem)

# Evaluate the bit error rate during hardware loop runtime
ber = BitErrorEvaluator(modem, modem)
loop.add_evaluator(ber)

# Add visualizations to be displayed during hardware loop runtime
loop.add_plot(ArtifactPlot(ber))
loop.add_plot(DeviceTransmissionPlot(device))
loop.add_plot(DeviceReceptionPlot(device))
loop.add_plot(ReceivedConstellationPlot(modem))

# Run the hardware loop
loop.num_drops = 100
loop.run()
