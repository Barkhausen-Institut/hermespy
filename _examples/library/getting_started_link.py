from hermespy import Scenario, Transmitter, Receiver
from hermespy.modem import WaveformGeneratorPskQam

transmitter = Transmitter()
transmitter.waveform_generator = WaveformGeneratorPskQam()

receiver = Receiver()
receiver.waveform_generator = WaveformGeneratorPskQam()

scenario = Scenario()
scenario.add_transmitter(transmitter)
scenario.add_receiver(receiver)

transmitted_signal, _ = transmitter.send()
propagated_signal, channel_state = scenario.channel(transmitter, receiver).propagate(transmitted_signal)

received_signal = receiver.receive(propagated_signal)
received_bits, received_symbols = receiver.demodulate(propagated_signal, channel_state)