=========================
Null Steering Beamformer
=========================

.. inheritance-diagram:: hermespy.beamforming.nullsteeringbeamformer
   :parts: 1

Null-steering :footcite:`nullsteeringbeamformer:zarifi` transmit beamformers aim to maximize the received signal power in the direction of the intended receiver while substantially 
reducing the power impinging on the unintended receivers located in other directions.

Let us introduce

.. math::
   \mathbf{a}_{\psi_{\bullet}} \triangleq
   \left[e^{j \frac{2\pi}{\lambda} r_1 \cos(\phi - \psi_1)}, \dots, e^{j \frac{2\pi}{\lambda} r_K \cos(\phi - \psi_K)} \right]^T

and

.. math::
   \mathbf{w} \triangleq
   \left[ w_1 \dots w_K \right]^T

Where :math:`\mathbf{a}_{\psi_{\bullet}}` is the array propagation matrix and the vector :math:`W` corresponds to the beamforming weights.
Let

.. math::
   \mathbf{A} \triangleq \left[ \mathbf{a}_{\phi_1} \dots \mathbf{a}_{\phi_L} \right]

where :math:`A` is the matrix that contains the array propagation vectors of all AoAs and :math:`P_T` denote the maximum admissible total transmission power.
The beamformer has maximized power in the intended direction and nulls at the unintended directions when:

.. math::
   \max_{\mathbf{w}} & \quad |\mathbf{w}^H \mathbf{a}_0|^2 \\
   \text{subject to} & \quad \mathbf{w}^H \mathbf{A} = 0 \\
                        & \quad \mathbf{w}^H \mathbf{w} \leq P_T.

The optimal solution to the above expression can be derived as:

.. math::
   \mathbf{w}_{\text{ns}} = \frac{\sqrt{P_T}}{\|(\mathbf{I} - \mathbf{P_A})\mathbf{a}_0\|} \cdot (\mathbf{I} - \mathbf{P_A})\mathbf{a}_0

where

.. math::
   \mathbf{P_A} \triangleq \mathbf{A}(\mathbf{A}^H\mathbf{A})^{-1}\mathbf{A}^H

is the orthogonal projection matrix onto the subspace spanned by the columns of :math:`A`.
As such, :math:`W_{\text{ns}}` is in fact the orthogonal projection of onto the null space of :math:`a_0`.
Thus, :math:`W_{\text{ns}}` is the null steering beamformer.
   
The radiation pattern of a NullSteeringBeamformer can be visualised using the following code snippet.
A uniform array is ctreated and the NullSteeringBeamformer is assigned to the array.
The radiation pattern can be visualised by specifiying the beamformer’s transmitting focus point and calling the plot_pattern method.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_radiation_pattern.py
   :language: python
   :linenos:
   :lines: 7-23 

.. image:: ../../images/nullsteeringbeamformer_radiation_pattern.png
   :width: 400

The following section provides an example for the usage of the nullsteeringbeamformer in a communication scenario.

For this a new simulation environment is initialised.
A digital antenna array consisting of ideal isotropic antenna elements spaced at half wavelength intervals is created and is assigned to a device representing a base station.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 7-20

To probe the characterestics a cosine waveform is generated as the transmit waveform.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 22-34

The base station device is configured to transmit the beamformed signal by assigning the NullSteeringBeamformer to the base station device.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 36-45

Now the simulation can be extended to evalaute the performance in a real world communication scenario.
For this three devices representing the UEs are added, to be illuminated by the BS.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 47-60

The user equipments can be configured to receive the signal by setting them up as receiving modems.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 61-70

Now as defined the nullsteeringbeamformer illuminates the maximum radiation on one UE whereas the nulls on the others.
This can be realised by configuring the transmit foucs of the Beamformer.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 71-79

The propgation characterestics between the BS and the UEs can be modelled using the SpatialDelayChannel Model.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 80-99

Now hermespy can be instructed to conduct a simulation campaign to evaluate the received signal power at the UEs by adding the evaluator.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteeringbeamformer_Tx.py
   :language: python
   :linenos:
   :lines: 101-118

In the previous example the NullSteeringBeamformer was assigned to transmitting base station.
The same example can be adpated to study the performance of the beamformer assigned to a receiving base station.

The base station is configured for reception.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteering_Rx.py
   :language: python
   :linenos:
   :lines: 37-46

The UEs will be configured as transmitting.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteering_Rx.py
   :language: python
   :linenos:
   :lines: 64-72

The receieve focus of the beamformer will be configured.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteering_Rx.py
   :language: python
   :linenos:
   :lines: 75-82

The performance of the beamformer is studied by analysing the received signal quality from the respective UEs.
For this purpose, the Error Vector Magnitude of the consetallation diagram of the Received signal is evaluated.

.. literalinclude:: ../../scripts/examples/beamforming_nullsteering_Rx.py
   :language: python
   :linenos:
   :lines: 105-122

.. autoclass:: hermespy.beamforming.nullsteeringbeamformer.NullSteeringBeamformer
   :private-members: _encode, _decode

.. footbibliography::
