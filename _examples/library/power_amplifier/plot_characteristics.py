import matplotlib.pyplot as plt
from hermespy.modem.rf_chain_models import PowerAmplifier, SalehPowerAmplifier, RappPowerAmplifier, ClippingPowerAmplifier,\
    CustomPowerAmplifier

saturation_amplitude = 1.0

power_amplifier = PowerAmplifier(saturation_amplitude=saturation_amplitude)
power_amplifier.plot()

rapp_power_amplifier = RappPowerAmplifier(saturation_amplitude=saturation_amplitude,
                                          smoothness_factor=0.5)
rapp_power_amplifier.plot()

clipping_power_amplifier = ClippingPowerAmplifier(saturation_amplitude=saturation_amplitude)
clipping_power_amplifier.plot()

saleh_power_amplifier = SalehPowerAmplifier(saturation_amplitude=saturation_amplitude,
                                            amplitude_alpha=1.9638,
                                            amplitude_beta=0.9945,
                                            phase_alpha=2.5293,
                                            phase_beta=2.8168)
saleh_power_amplifier.plot()

plt.show()
