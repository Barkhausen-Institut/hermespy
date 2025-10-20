================
Synchronization
================

Synchronization between simulated devices is modeled by :doc:`Trigger Models<synchronization.TriggerModel>`.
Simulated device instances sharing the same :doc:`Trigger Model<synchronization.TriggerModel>` instance
are considered time-synchronized with each other, but not with other simulated devices that do not share the same :doc:`Trigger Model<synchronization.TriggerModel>`.

.. mermaid::
   :align: center

   %%{init: {"flowchart":{"useMaxWidth": false}}}%%
   flowchart LR

   subgraph Synchronization
      direction TB
      trigger[TriggerModel] --> trigger_realization[TriggerRealization]
   end

   device_a[Simulated Device]
   device_b[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&vellip;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]:::invis
   device_c[Simulated Device]

   trigger_realization --> device_a
   trigger_realization --> device_b
   trigger_realization --> device_c

   classDef invis fill-opacity:0,stroke-opacity:0,font-weight:bold;

   click trigger href "simulation.synchronization.TriggerModel.html"
   click trigger_realization href "simulation.synchronization.TriggerRealization.html"
   click device_a href "simulation.simulated_device.html"
   click device_b href "simulation.simulated_device.html"
   click device_c href "simulation.simulated_device.html"

During the generation of each simulation drop, the :doc:`Trigger Model<synchronization.TriggerModel>` instance
is realized once for all simulated devices sharing the same instance, resulting in a single :doc:`Trigger Realization<synchronization.TriggerRealization>`.

Note that, as the name suggests, HermesPy's synchronization model is considered to be at the trigger level of
transmitting and receiving devices, meaning propagation delays due to channel models linking synchronized devices
must still be compensated by appropriate equalization routines.
The currently available :doc:`Trigger Model<synchronization.TriggerModel>` implementations are:

.. include:: synchronization._table.rst

Consider a scenario featuring four wireless devices, with two devices respectively linked, interfering with each other on partially
overlapping frequency bands.

.. literalinclude:: /scripts/examples/simulation_synchronization.py
   :language: python
   :linenos:
   :lines: 14-17

The exchanged waveforms are identical, however, devices are considered to be only synchronized to their linked partners.

.. literalinclude:: /scripts/examples/simulation_synchronization.py
   :language: python
   :linenos:
   :lines: 38-44

Of course, the abstract *TriggerModel* in the above snippet must be replaced by the desired implementation from the list above.
By generating a single drop of the simulation and plotting the bit error rates of the two devices,
we may visualize the impact of partially overlapping commuication frames in time-domain due to the interference
in between the two links.

.. literalinclude:: /scripts/examples/simulation_synchronization.py
   :language: python
   :linenos:
   :lines: 46-55

.. toctree::
   :hidden:

   synchronization.TriggerModel
   synchronization.TriggerRealization
   synchronization.StaticTrigger
   synchronization.RandomTrigger
   synchronization.SampleOffsetTrigger
   synchronization.TimeOffsetTrigger
