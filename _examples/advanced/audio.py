# -*- coding: utf-8 -*-
# This example showcases how HermesPy's hardware loop functionality
# can be configured to transmit and receive OFDM-modulated signals
# usinge the sound card of a computer.

from hermespy.hardware_loop import (
    HardwareLoop,
    AudioScenario,
    AudioDevice,
)
from hermespy.modem import (
    BitErrorEvaluator,
    DuplexModem,
    OFDMWaveform,
    OrthogonalLeastSquaresChannelEstimation,
    OrthogonalZeroForcingChannelEqualization,
    SchmidlCoxSynchronization,
    SchmidlCoxPilotSection,
    GridResource,
    GridElement,
    ElementType,
    PrefixType,
    SymbolSection,
    GuardSection,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Create a new hardware loop
loop = HardwareLoop[AudioScenario, AudioDevice](AudioScenario())

# Connect to the first sound card
soundcard = loop.new_device(
    playback_device=6,
    record_device=4,
    playback_channels=[1],
    record_channels=[1],
    max_receive_delay=1.0,
)

# Configure a transmit and receive modem at the soundcard
modem = DuplexModem()
modem.waveform = OFDMWaveform(
    modulation_order=2,
    dc_suppression=False,
    num_subcarriers=4096,
    channel_estimation=OrthogonalLeastSquaresChannelEstimation(),
    channel_equalization=OrthogonalZeroForcingChannelEqualization(),
    synchronization=SchmidlCoxSynchronization(),
    grid_resources=[
        GridResource(1, PrefixType.CYCLIC, 0.0703125, [
            GridElement(ElementType.NULL, 600),
            GridElement(ElementType.DATA, 2100),
            GridElement(ElementType.NULL, 600),
        ]),
        GridResource(1, PrefixType.CYCLIC, 0.078125, [
            GridElement(ElementType.NULL, 600),
            GridElement(ElementType.DATA, 2100),
            GridElement(ElementType.NULL, 600),
        ]),
    ],
    grid_structure=[
        SchmidlCoxPilotSection(),
        SymbolSection(1, [1, 0, 0, 0]),
        GuardSection(35.677083e-6),
        SymbolSection(1, [1, 0, 0, 0]),
        GuardSection(35.677083e-6),
    ],
)
soundcard.add_dsp(modem)

# Add a bit error evaluation to the hardware loop
loop.add_evaluator(BitErrorEvaluator(modem, modem))

# Activate manual confirmation for each drop
loop.manual_triggering = True

# Start the hardware loop
loop.num_drops = 10
loop.run()
