==========
Simulation
==========

The simulation module provides the basis for all HermesPy simulations,
including noise and hardware modeling.

It is design around the Ray framework for parallel and distributed computing,
enabling the efficient parallelization of physical layer Monte Carlo simulations scaling to
multicore-CPUs, computing clusters and cloud computing environments.
Every Monte-Carlo physical layer simulation campaign is composed of a set of
parallel numeric physical layer simulations, each of which is individually
parameterized.

.. include:: ../flowcharts/simulation_parallel_monte_carlo.rst

During execution, a central simulation controller will distribute parameter combinations
selected from the simulation parameter grid to the individual physical layer simulation instances
and collect the resulting performance evaluations.
Each of the physical layer simulations is executed in a separate Python
thread and distributed to the available CPU / GPU and memory resources by Ray.

The physical layer in turn, at its core, is composed of simulated devices and
reciprocal channels interconnecting the physical devices.

.. include:: ../flowcharts/simulated_physical_layer.rst

Simulated devices represent any physical entity capabable of transmitting or receiving electromagnetic waveforms.
This can be on the one hand communication devices such base-stations, smartphones, wifi routers, laptops or other Internet-of-things devices.
On the other hand, they may be sensing devices such as automotive FMCW radars and bistatic radar transmitters and receivers.
In general, devices can be aribtrarily positioned and oriented in space and transmit and/or receive arbitrary electromagnetic waveforms at arbitrary carrier frequencies.

The physical properties of devices' front-end hardware can be specified in detail to allow for a more realistic simulation of the hardware effects on
the transmitted and received signals.
This includes modeling analog to digital conversion (ADC) and digital to analog conversion (DAC) introducing quantization noise and non-linearities,
oscillator effects such as in-phase/quadrature imbalance (I/Q) and phase noise (PN),
amplification non-linearities in the power amplifier (PA) and low-noise amplifier (LNA),
mutual coupling between individual antennas in antenna arrays (MC),
polarimetric radiation patterns of individual antennas in antenna arrays (ANT),
and transmit-receive isolation between transmitting and receiving antennas in antenna arrays of duplex devices (ISO).

.. include:: ../flowcharts/device_rf_interaction.rst

Considering a single simulation iteration, within each simulated device,
a digital base-band signal model to be transmitted is generated and
sequentially passed through the radio-frequency transmission chain (RF Tx) models.
The emerging analog signal is then duplicated, with one copy being passed through the mutual coupling
and antenna models and the other copy being passed throug the transmit-receive isolation model.
The signal copy considering mutual coupling and antenna effects is then propagated over all channel models
linking the specific device to all other devices in the simulation.
The received signals, after propgation over the channel models, are then once again passed through the
antenna and mutual coupling models.
Before considering the effects of the radio-frequency reception chain (RF Rx) models, the received signals
are superimposed with the signal copy considering transmit-receive isolation.
Finally, the received signal is passed through the RF Rx models and the resulting digital base-band signal
is processed by the configured receive digital signal processing.

.. note::

   The order of hardware-impairment models is currently fixed and cannot be changed.
   This system is under active development and will be expanded in the future to enable
   arbitrary ordering of hardware-impairment models and thus more accurate representations of custom front-ends.

Configuring the simulation exactly as the above graphs show
is as simple as:

.. literalinclude:: ../scripts/examples/simulation.py
   :language: python
   :linenos:
   :lines: 9-29

In practice, limiting the number of parallel simulations by setting the ``num_actors`` property
is usually not required, as Ray will automatically scale the number of parallel simulations.
However, sometimes memory constraints may require setting a lower value.

Typical parameters to be varied in a Monte-Carlo simulation are the receive signal-to-noise ratio and the carrier frequency
of the exchanged signals.
For the specific values of an SNR between 0 and 20 dB and carrier frequencies of 1, 10 and 100 GHz, the overall simulation
parameter grid may look as follows:

+------------------------+-------------------------------------------+--------------------------------------------+---------------------------------------------+
|                        | Carrier-Frequency                                                                                                                    |
+------------------------+-------------------------------------------+--------------------------------------------+---------------------------------------------+
| SNR                    |  :math:`1~\mathrm{GHz}`                   | :math:`10~\mathrm{GHz}`                    | :math:`100~\mathrm{GHz}`                    |
+========================+===========================================+============================================+=============================================+
| :math:`0~\mathrm{dB}`  | :math:`(\ 0~\mathrm{dB}, 1~\mathrm{GHz})` | :math:`(\ 0~\mathrm{dB}, 10~\mathrm{GHz})` | :math:`(\ 0~\mathrm{dB}, 100~\mathrm{GHz})` |
+------------------------+-------------------------------------------+--------------------------------------------+---------------------------------------------+
| :math:`10~\mathrm{dB}` | :math:`(10~\mathrm{dB}, 1~\mathrm{GHz})`  | :math:`(10~\mathrm{dB}, 10~\mathrm{GHz})`  | :math:`(10~\mathrm{dB}, 100~\mathrm{GHz})`  |
+------------------------+-------------------------------------------+--------------------------------------------+---------------------------------------------+
| :math:`20~\mathrm{dB}` | :math:`(20~\mathrm{dB}, 1~\mathrm{GHz})`  | :math:`(20~\mathrm{dB}, 10~\mathrm{GHz})`  | :math:`(20~\mathrm{dB}, 100~\mathrm{GHz})`  |
+------------------------+-------------------------------------------+--------------------------------------------+---------------------------------------------+

The simulation class provides a convenient interface to sweep over virtually any class property of any class contained within the simulation.
In the above example, the carrier frequency is configured by setting the :attr:`carrier_frequency<hermespy.simulation.simulated_device.SimulatedDevice>` property
of the simulated devices.
Since the SNR is a frequently used parameter, it has a shorthand and can be globally set for all devices within the simulation without the need to
specify the device class:

.. literalinclude:: ../scripts/examples/simulation.py
   :language: python
   :linenos:
   :lines: 21-35

Now, by default, devices won't transmit or receive any signals.
For the sake of simplicity in this example, we will configure the devices to transmit and receive
:math:`100` samples of a random complex signal with an amplitude of :math:`1` and a bandwith of :math:`100~\mathrm{MHz}`.

.. literalinclude:: ../scripts/examples/simulation.py
   :language: python
   :linenos:
   :lines: 37-48

Even though the simulation is now fully configured in terms of its basic physical layer description
and the parameter grid, no performance evaluations will be generated when executing the simulation.
Instead, each performance indicator to be evaluated must be explicitly configured.
Multiple modules of HermesPy provide implementations of performance indicator evaluators for
performance indicators relevant to the respective module's topic.
In this example, one of the most basic performance indicators is the signal power received by each device:

.. literalinclude:: ../scripts/examples/simulation.py
   :language: python
   :linenos:
   :lines: 50-52

The simulation can be executed by calling the :meth:`run<hermespy.simulation.simulation.Simulation.run>` method.
All configured performance indicators will be evaluated for each parameter combination and the results
returned in a :class:`MonteCarloResult<hermespy.core.monte_carlo.MonteCarloResult>`.
From there, the result can be printed to the console, plotted, or saved to the drive.

.. literalinclude:: ../scripts/examples/simulation.py
   :language: python
   :linenos:
   :lines: 54-62

.. toctree::
   :hidden:
   :maxdepth: 1
   
   simulation.simulation.Simulation
   simulation.simulated_device
   simulation.animation
   simulation.antennas
   simulation.noise
   simulation.synchronization
   simulation.rf_chain.adc
   simulation.rf_chain
   simulation.rf_chain.phase_noise
   simulation.rf_chain.amplifier
   simulation.coupling
   simulation.isolation
   simulation.simulation.SimulationScenario
   simulation.simulation.SimulatedDrop
