=================================
Radio-Frequency Hardware Modeling
=================================

HermesPy's simulation module includes a  block-based modeling framework for arbitrary radio-frequency front-ends.
It provides a flexible and extensible platform to design, simulate and analyze
the effects of radio-frequency design choices on the overall system performance.

Individual radio-frequency blocks exchange :class:`~hermespy.simulation.rf.signal.RFSignal` models in a feed-forward fashion.
Loops are currently not supported.
The DSP layer is fed from, and fed by, the RF layer through :class:`~hermespy.simulation.rf.block.DSPInputBlock` and :class:`~hermespy.simulation.rf.block.DSPOutputBlock`, respectively.
Signals not feed to blocks are considered inputs to the antennas during transmission, while unused block input ports are exposed as antenna ports during reception.
An example of a simple front-end chain is illustrated below:

.. mermaid::

   flowchart LR
      dsp_in[DSPInputBlock] --> sig_a{{RFSignal}} --> rf_a[RFBlock] --> sig_b{{RFSignal}} --> rf_b[RFBlock] --> sig_c{{RFSignal}};
      rf_in[RFBlock] --> sig_d{{RFSignal}} --> rf_b;
      sig_e{{RFSignal}} --> rf_c[RFBlock] --> sig_f{{RFSignal}} --> dsp_out[DSPOutputBlock];
      sig_d --> rf_c;

      click dsp_in "block.html#hermespy.simulation.rf.block.DSPInputBlock";
      click dsp_out "block.html#hermespy.simulation.rf.block.DSPOutputBlock";
      click rf_in,rf_a,rf_b,rf_c "block.html#hermespy.simulation.rf.block.RFBlock";
      click sig_a,sig_b,sig_c,sig_d,sig_e,sig_f "signal.html#hermespy.simulation.rf.signal.RFSignal";

It can be configured by interacting with instances of :class:`~hermespy.simulation.rf.chain.RFChain`,
which provides a high-level interface to manage the blocks and their interconnections:

.. literalinclude:: ../../../scripts/examples/simulation_rf.py
   :language: python
   :linenos:
   :lines: 15-31

Afterwards the radio-frequency chain model can be assigend to devices by setting the :attr:`~hermespy.simulation.simulated_device.SimulatedDevice.rf` property.
Note that changing the chain model might require updating the antenna array configuration to ensure that the number of antenna ports matches the number of signal inputs and outputs.

.. literalinclude:: ../../../scripts/examples/simulation_rf.py
   :language: python
   :linenos:
   :lines: 33-35

The following presets modeling commercially availabe hardware are currently available:

.. include:: presets/presets._table.rst

.. toctree::
   :hidden:
   :maxdepth: 1

   chain
   block
   blocks/index
   presets/index
   noise/index
   signal