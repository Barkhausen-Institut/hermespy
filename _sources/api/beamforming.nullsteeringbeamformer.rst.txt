=========================
Null Steering Beamformer
=========================

.. inheritance-diagram:: hermespy.beamforming.nullsteeringbeamformer
   :parts: 1

The following section provides an example for the usage of the nullsteeringbeamformer. 
A digital antenna array consisting of ideal isotropic antenna elements spaced at half wavelength intervals is created in the following code snippet.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 13-23

To investigate how the radiation pattern of the antenna array changes with respect to its topology.
For this a new simulation environment is initialised and the custom antenna array is assigned to a device representing a transmitting base station.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 25-33

To probe the characterestics a cosine waveform is generated as the transmit waveform.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 35-45

The nullsteeringbeamformer is assigned to the custom array to transmit beamformed probing signal.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 47-52

The radiation pattern of the array can be visualised by specifiying the beamformer’s transmitting focus point and calling the plot_pattern method.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer_radiation_pattern.py
   :language: python
   :linenos:
   :lines: 27-29

.. image:: ../images/nullsteeringbeamformer_radiation_pattern.PNG
   :width: 400

Now the simulation can be extended to evalaute the performance in a real world communication scenario.
For this three devices representing the UEs are added, to be illuminated by the BS.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 54-73

Now as defined the nullsteeringbeamformer illuminates the maximum radiation on one UE whereas the nulls on the others.
This can be realised by configuring the transmit foucs of the Beamformer.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 75-78

The propgation characterestics between the BS and the UEs can be modelled using the SpatialDelayChannel Model.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 80-97

Now hermespy can be instructed to conduct a simulation campaign to evaluate the received signal power at the UEs by adding the evaluator.

.. literalinclude:: ../scripts/examples/beamforming_nullsteeringbeamformer.py
   :language: python
   :linenos:
   :lines: 99-105

.. autoclass:: hermespy.beamforming.nullsteeringbeamformer.NullSteeringBeamformer
   :private-members: _encode

.. footbibliography::
