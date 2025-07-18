# HermesPy

HermesPy (Heterogeneous Radio Mobile Simulator - Python) is a semi-static link-level simulator based on time-driven mechanisms.

It provides a framework for the link-level simulation of multi-RAT wireless sensing and communication links, consisting of multiple transmit and receive devices, which may operate at different carrier frequencies and bandwidths. 
Digital signal processing algorithms implemented in HermesPy's API can be investigated within full-scale Monte-Carlo simulation campaigns deployable to high-performance computing clusters or applied directly within software defined radio testbeds.

For detailed a full API documentation, please consult the official
[documentation](https://hermespy.org/index.html) website.

The project is completely open-source and published under the [GNU AGPL License](https://github.com/Barkhausen-Institut/hermespy/blob/master/LICENSE) on [GitHub](https://github.com/Barkhausen-Institut/hermespy/).
Please cite [Adler et al](https://ieeexplore.ieee.org/document/9950269) for any results obtained with the help of HermesPy.
Contributions are highly welcome and can be made by means of [GitHub pull requests](https://github.com/Barkhausen-Institut/hermespy/pulls).
For issue reports and feature requests please open a new [GitHub issue](https://github.com/Barkhausen-Institut/hermespy/issues)
or directly contact the current [maintainer](https://www.linkedin.com/in/jan-adler/).

## Installation

The recommended installation method for end-users of HermesPy is pulling the pre-built wheels from [PyPI](https://pypi.org/project/hermespy/) by executing
```
python -m pip install hermespy
```
in a Python environment of version 3.10 to 3.12.
The current development version can be cloned from [GitHub](https://github.com/Barkhausen-Institut/hermespy) and installed as an editable by executing
```
git clone --recursive git@github.com:Barkhausen-Institut/hermespy.git
cd hermespy
python -m pip install -v -e .
```

Detailed installation instructions as well as a list of optional feature flags can be found in the respective [documentation section](https://hermespy.org/installation.html).

## Quick Start

The following examples provide a starting point for understanding HermesPy's high-level API.
There are three modules providing digital signal processing algorithms: [Modem for communications](https://hermespy.org/api/modem/index.html), [radar for sensing](https://hermespy.org/api/radar/index.html) and [jcas for joint applications](https://hermespy.org/api/jcas/index.html).

```python
from hermespy.modem import ReceivingModem, RRCWaveform, BitErrorEvaluator
from hermespy.radar import Radar, FMCW
from hermespy.jcas import MatchedFilterJcas

# Initialize a DSP algorithm from each module
com = ReceivingModem(waveform=RRCWaveform())
sense = Radar(waveform=FMCW(num_chirps=100, pulse_rep_interval=3e-6))
joint = MatchedFilterJcas(10.0, waveform=RRCWaveform())

# Initialize an exemplary KPI evaluation routine
ber = BitErrorEvaluator(joint, com, plot_surface=False)
```

Digital signal processing algorithms and their respective evaluation routines may be deployed to simulation campaigns by assigning them to device models initialized from the simulation module.
API users can configure sweeps over parameters of interest during simulation runtime by assigning multiple values to any object property via the new dimension interface.

```python
import matplotlib.pyplot as plt
from hermespy.core import dB
from hermespy.simulation import Simulation

# Initialize a simulation running each DSP algorithm on a dedicated device
sim = Simulation(seed=42)
sim.new_device(carrier_frequency=26e9).add_dsp(sense)
sim.new_device(carrier_frequency=1e9).add_dsp(com)
sim.new_device(carrier_frequency=1e9).add_dsp(joint)

# Configure the simulation to run the KPI evaluation routine
sim.add_evaluator(ber)

# Configure the simulation to sweep over two parameter dimensions
sim.new_dimension('noise_level', dB(-20, -10, 0, 10, 20, 30), *sim.devices)
sim.new_dimension('carrier_frequency', [1e9, 13e9, 26e9], sim.devices[0], title="CF")

# Run the simulation and plot the results
sim.num_samples = 200
sim.run().plot()
plt.show()
```

The same evaluations carried out within distributed simulation campaigns can be deployed within real hardware testbeds by assigning digital signal processing algorithms and evaluation routines to devices initialized from the hardware loop module instead.

```python
import matplotlib.pyplot as plt
from hermespy.hardware_loop import HardwareLoop, PhysicalScenarioDummy, ReceivedConstellationPlot

# Initialize a hardware loop running each DSP algorithm on a dedicated device
loop = HardwareLoop(PhysicalScenarioDummy(seed=42))
loop.new_device(carrier_frequency=1e9).add_dsp(sense)
loop.new_device(carrier_frequency=1e9).add_dsp(com)
loop.new_device(carrier_frequency=1e9).add_dsp(joint)

# Configure the hardware loop to run the KPI evaluation routine
loop.add_evaluator(ber)

# Configure the hardware loop to sweep over a parameter dimension
loop.new_dimension('power', dB(-30, -20, -10, 0, 10), loop.scenario.devices[0], title="Interference Power")

# Configure the hardware loop to visualize a received signal
loop.add_plot(ReceivedConstellationPlot(com, 'Rx Constellation'))

# Run the hardware loop and plot the results
loop.num_drops = 10
loop.run().plot()
plt.show()
```

More examples like this can be found in the following locations:

| Link | Description |
| ---- | ----------- |
| [Getting Started](https://github.com/Barkhausen-Institut/hermespy/tree/master/_examples/getting_started) | Introductory examples for inexperienced users |
| [Advanced](https://github.com/Barkhausen-Institut/hermespy/tree/master/_examples/advanced) | Examples for advanced users |
| [Notebooks](https://hermespy.org/tutorials.html) | Full Jupyter notebooks for advanced users and developers |
| [Snippets](https://github.com/Barkhausen-Institut/hermespy/tree/master/docssource/scripts/examples) | Executable API snippets shown throughout the documentation |
