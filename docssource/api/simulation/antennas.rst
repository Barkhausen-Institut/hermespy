=========
Antennas
=========

The simulation module extends the core module's antenna descriptions
by modeling radio-frequency chains connected to antenna arrays and 
directive polarization / gain characteristics of individual antennna elements within the array.

Simulated antenna arrays are described by the :class:`SimulatedAntennaArray<hermespy.simulation.antennas.SimulatedAntennaArray>` class.
On the surface, this description contains a combination of a set of available :class:`SimulatedAntennaPorts<hermespy.simulation.antennas.SimulatedAntennaPort>`,
to which :class:`RfChains<hermespy.simulation.rf_chain.rf_chain.RfChain>` feeding one or multiple :class:`SimulatedAntenna<hermespy.simulation.antennas.SimulatedAntenna>` elements are connected.
In a fully digitally controlled antenna array, each port feeds a single antenna element over a dedicated RF-chain:

.. mermaid::
   :align: center

   %%{init: {"flowchart":{"useMaxWidth": false}}}%%
   flowchart LR

   porta[Antenna Port] --> rfa[RF-Chain] --> anta[Antenna]
   portb[Antenna Port] --> rfb[RF-Chain] --> antb[Antenna]
   x[&vellip;]:::invis ~~~ b[&vellip;]:::invis ~~~ c[&vellip;]:::invis
   portc[Antenna Port] --> rfc[RF-Chain] --> antc[Antenna]

   classDef invis fill-opacity:0,stroke-opacity:0,font-weight:bold;

   click porta "antennas.SimulatedAntennaPort.html" "Simulated Antenna Port"
   click portb "antennas.SimulatedAntennaPort.html" "Simulated Antenna Port"
   click portc "antennas.SimulatedAntennaPort.html" "Simulated Antenna Port"
   click rfa "rf_chain.html" "RF-Chain"
   click rfb "rf_chain.html" "RF-Chain"
   click rfc "rf_chain.html" "RF-Chain"
   click anta "antennas.SimulatedAntenna.html" "Simulated Antenna"
   click antb "antennas.SimulatedAntenna.html" "Simulated Antenna"
   click antc "antennas.SimulatedAntenna.html" "Simulated Antenna"

Manually defining an antenna array this way can be achieved by instantiating the :class:`SimulatedCustomArray<hermespy.simulation.antennas.SimulatedCustomArray>` class
and individually adding the desired antenna elements, which will automatically be connected to a new RF-chain and antenna port:

.. literalinclude:: ../../scripts/examples/simulation_antennas.py
   :language: python
   :linenos:
   :lines: 13-17

In this example, :class:`SimulatedPatchAntennas<hermespy.simulation.antennas.SimulatedPatchAntenna>` are uniformly distributed
along the array's x-axis, with a spacing of 0.5 wavelengths, effectively creating a uniform linear array.
The antenna elements are connected to a :class:`RfChain<hermespy.simulation.rf_chain.rf_chain.RfChain>`, respectively.
Since this is a rather common antenna configuration, the shorthand :class:`SimulatedUniformArray<hermespy.simulation.antennas.SimulatedUniformArray>` class
can be used to create the same antenna array:

.. literalinclude:: ../../scripts/examples/simulation_antennas.py
   :language: python
   :linenos:
   :lines: 22

In this case, instead of patch antennas, the array is populated with ideal isotropic antennas.
When simulating analog or hybrid antenna arrays, a single RF-chain can feed multiple antenna elements.

.. mermaid::
   :align: center

   %%{init: {"flowchart":{"useMaxWidth": false}}}%%
   flowchart LR

   porta[Antenna Port] --> rfa[RF-Chain] --> antaa[Antenna] & antab[&vellip;]:::invis & antac[Antenna]

   classDef invis fill-opacity:0,stroke-opacity:0,font-weight:bold;

   click porta "antennas.SimulatedAntennaPort.html" "Simulated Antenna Port"
   click rfa "rf_chain.html" "RF-Chain"
   click antaa "antennas.SimulatedAntenna.html" "Simulated Antenna"
   click antac "antennas.SimulatedAntenna.html" "Simulated Antenna"

This can be modeled by assigning multiple antenna elements to the same antenna port:

.. literalinclude:: ../../scripts/examples/simulation_antennas.py
   :language: python
   :linenos:
   :lines: 26-35

The snippet initializes a antenna array featuring 10 antenna ports, each feeding 5 antenna elements,
so that the array is populated with 50 antenna elements in total.
Within the context of a full simulation, antenna arrays are assigned as a configuration property
to :class:`SimulatedDevices<hermespy.simulation.simulated_device.SimulatedDevice>`:

.. literalinclude:: ../../scripts/examples/simulation_antennas.py
   :language: python
   :linenos:
   :lines: 39-41

.. toctree::
   :hidden:
   :maxdepth: 2

   antennas.SimulatedAntennaArray
   antennas.SimulatedAntenna
   antennas.SimulatedAntennaPort

.. footbibliography::
