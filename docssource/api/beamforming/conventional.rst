========================
Conventional Beamformer
========================

.. inheritance-diagram:: hermespy.beamforming.ConventionalBeamformer
   :parts: 1
   
Also refferred to as delay and sum beamformer.

The Bartlett\ :footcite:`1950:bartlett` beamformer, also known as conventional or delay and sum beamformer, maximizes the power transmitted or received towards a single direction of interest :math:`(\theta, \phi)`, 
where :math:`\theta` is the zenith and :math:`\phi` is the azimuth angle of interest in spherical coordinates, respectively.

Let :math:`\mathbf{X} \in \mathbb{C}^{N \times T}` be the the matrix of :math:`T` time-discrete samples acquired by an antenna arrary featuring :math:`N` antennas.
The antenna array's response towards a source within its far field emitting a signal of small relative bandwidth is :math:`\mathbf{a}(\theta, \phi) \in \mathbb{C}^{N}`.
Then

.. math::

   \hat{P}_{\mathrm{Conventional}}(\theta, \phi) = \mathbf{a}^\mathsf{H}(\theta, \phi)  \mathbf{X} \mathbf{X}^\mathsf{H} \mathbf{a}(\theta, \phi)

is the Conventional beamformer's power estimate     with

.. math::

   \mathbf{w}(\theta, \phi) = \mathbf{a}(\theta, \phi)

being the beamforming weights to steer the sensor array's receive characteristics towards direction :math:`(\theta, \phi)`, so that

.. math::

   \mathcal{B}\lbrace \mathbf{X} \rbrace = \mathbf{w}^\mathsf{H}(\theta, \phi) \mathbf{X}

is the implemented beamforming equation.
   

The following section provides an example for the usage of the Conventional Beamformer in a communication scenario.

For this a new simulation environment is initialised.
A digital antenna array consisting of ideal isotropic antenna elements spaced at half wavelength intervals is created and is assigned to a device representing a base station.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 7-20

To probe the characterestics a cosine waveform is generated as the transmit waveform.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 22-34

The base station device is configured to transmit the beamformed signal by assigning the Conventional Beamformer to the base station device.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 36-45

Now the simulation can be extended to evalaute the performance in a real world communication scenario.
For this two devices representing the UEs are added, to be illuminated by the BS.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 47-55

The user equipments can be configured to receive the signal by setting them up as receiving modems.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 57-63

Now as defined the Conventional Beamformer illuminates the maximum radiation on one UE.
This can be realised by configuring the transmit foucs of the Beamformer.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 65-70

The propgation characterestics between the BS and the UEs can be modelled using the SpatialDelayChannel Model.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 72-85

Now hermespy can be instructed to conduct a simulation campaign to evaluate the received signal power at the UEs by adding the evaluator.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Tx.py
   :language: python
   :linenos:
   :lines: 87-100

In the previous example the Conventional Beamformer was assigned to transmitting base station.
The same example can be adpated to study the performance of the beamformer assigned to a receiving base station.

The base station is configured for reception.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Rx.py
   :language: python
   :linenos:
   :lines: 36-45

The UEs will be configured as transmitting.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Rx.py
   :language: python
   :linenos:
   :lines: 58-64

The receieve focus of the beamformer will be configured.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Rx.py
   :language: python
   :linenos:
   :lines: 67-72

The performance of the beamformer is studied by analysing the received signal quality from the respective UEs.
For this purpose, the Error Vector Magnitude of the consetallation diagram of the Received signal is evaluated.

.. literalinclude:: ../../scripts/examples/beamforming_conventional_Rx.py
   :language: python
   :linenos:
   :lines: 89-103

.. automodule:: hermespy.beamforming.conventional
   :private-members: _decode, _encode

.. footbibliography::
