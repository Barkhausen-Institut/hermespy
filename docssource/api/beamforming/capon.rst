=================
Capon Beamformer
=================

.. inheritance-diagram:: hermespy.beamforming.CaponBeamformer
   :parts: 1

Capon beamformer, also referred to as Minimum Variance Distortionless Response (MVDR).

The Capon\ :footcite:`1969:capon` beamformer estimates the power :math:`\hat{P}` received from a direction :math:`(\theta, \phi)`, where :math:`\theta` is the zenith and :math:`\phi`  is the azimuth angle of interest in spherical coordinates, respectively.
Let :math:`\mathbf{X} \in \mathbb{C}^{N \times T}` be the the matrix of :math:`T` time-discrete samples acquired by an antenna arrary featuring :math:`N` antennas and

.. math::

   \mathbf{R}^{-1} = \left( \mathbf{X}\mathbf{X}^{\mathsf{H}} + \lambda \mathbb{I} \right)^{-1}

be the respective inverse sample correlation matrix loaded by a factor :math:`\lambda \in \mathbb{R}_{+}`.
The antenna array's response towards a source within its far field emitting a signal of small relative bandwidth is :math:`\mathbf{a}(\theta, \phi) \in \mathbb{C}^{N}`.
Then, the Capon beamformer's spatial power response is defined as

.. math::

   \hat{P}_{\mathrm{Capon}}(\theta, \phi) = \frac{1}{\mathbf{a}^{\mathsf{H}}(\theta, \phi) \mathbf{R}^{-1} \mathbf{a}(\theta, \phi)}

with

.. math::

   \mathbf{w}(\theta, \phi) = \frac{\mathbf{R}^{-1} \mathbf{a}(\theta, \phi)}{\mathbf{a}^{\mathsf{H}}(\theta, \phi) \mathbf{R}^{-1} \mathbf{a}(\theta, \phi)} \in \mathbb{C}^{N}

being the beamforming weights to steer the sensor array's receive characteristics towards direction :math:`(\theta, \phi)`, so that

.. math::

   \mathcal{B}\lbrace \mathbf{X} \rbrace = \mathbf{w}^\mathsf{H}(\theta, \phi) \mathbf{X}

is the implemented beamforming equation.
    

The following section provides an example for the usage of the Conventional Beamformer in a communication scenario.

For this a new simulation environment is initialised. 
A digital antenna array consisting of ideal isotropic antenna elements spaced at half wavelength intervals is created and is assigned to a device representing a base station.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 16-20 

To probe the characterestics a cosine waveform is generated as the transmit waveform.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 23-32

The base station device is configured to receive the signal from desired UE by assigning the Capon Beamformer to the base station device.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 39-43

Now the simulation can be extended to evaluate the performance in a real world communication scenario.
For this two devices representing the UEs are added, to be illuminated by the BS.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 46-53

The user equipments are configured to transmit the probing signal.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 58-61

Now as defined the Capon Beamformer focuses on one UE. 
This can be realised by configuring the receive foucs of the Beamformer.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 66-68

The propgation characterestics between the BS and the UEs can be modelled using the SpatialDelayChannel Model.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 73-83

The performance of the beamformer is studied by analysing the received signal quality from the respective UEs.
For this purpose, the Error Vector Magnitude of the consetallation diagram of the Received signal is evaluated.

.. literalinclude:: ../../scripts/examples/beamforming_capon.py
   :language: python
   :linenos:
   :lines: 88-89

.. automodule:: hermespy.beamforming.capon
   :private-members: _decode
.. footbibliography::
