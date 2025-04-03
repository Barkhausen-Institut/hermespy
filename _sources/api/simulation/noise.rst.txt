======
Noise
======

.. inheritance-diagram:: hermespy.simulation.noise.model.NoiseModel hermespy.simulation.noise.level.NoiseLevel hermespy.simulation.noise.model.NoiseRealization
   :parts: 1

Hardware noise is one of the primary factors limiting the performance of wireless systems in both communication and sensing applications.
While any processing step introduces noise to some extent, Hermes currently assumes receiving front-ends to be the primary source of noise power added to any incoming signal.

HermesPy's noise configuration consists of classes,
the :class:`NoiseModel<hermespy.simulation.noise.model.NoiseModel>` representing the noise's statistical properties and
the :class:`NoiseLevel<hermespy.simulation.noise.level.NoiseLevel>` representing the noise's power level.

When generating a new realization of the statistical :class:`NoiseModel<hermespy.simulation.noise.model.NoiseModel>`,
the expected power returned by the :class:`NoiseLevel's<hermespy.simulation.noise.level.NoiseLevel>`
:meth:`get_power()<hermespy.simulation.noise.level.NoiseLevel.get_power>` method is passed to the
:meth:`realize<hermespy.simulation.noise.model.NoiseModel.realize>` method
to obtain a :class:`NoiseRealization<hermespy.simulation.noise.model.NoiseRealization>` instance.

.. mermaid::
   :align: center

   flowchart LR

   model[NoiseModel] -->|realize| realization[NoiseRealization]
   level[NoiseLevel] -->|get_power| realization

The currently available implementations of noise models are

.. list-table::
   :header-rows: 1

   * - Model
     - Realization
     - Description

   * - :doc:`noise.model.AWGN`
     - :doc:`noise.model.AWGNRealization`
     - Complex additive white Gaussian noise.


The currently available noise levels

.. list-table::
   :header-rows: 1

   * - Level
     - Description

   * - :doc:`noise.level.N0`
     - Noise floor in terms of watts.
   * - :doc:`noise.level.SNR`
     - Signal-to-noise ratio with respect to a reference.
   * - :doc:`modem.noise.EBN0`
     - Bit energy to noise power ratio with respect to a waveform.
   * - :doc:`modem.noise.ESN0`
     - Symbol energy to noise power ratio with respect to a waveform.


Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>` thermal
noise model is as simple as setting its :attr:`noise<hermespy.simulation.simulated_device.SimulatedDevice.noise_model>` porperty to an instance of a noise model.

.. literalinclude:: ../../scripts/examples/simulation_noise.py
   :language: python
   :linenos:
   :lines: 6-11

Of course, the abstract *Noise* model in the above snippet has to be replaced with a specific implementation
from the above table.
The actual noise level can be adjusted indepentely from the model by either setting the default noise level property.

.. literalinclude:: ../../scripts/examples/simulation_noise.py
   :language: python
   :linenos:
   :lines: 13-17

If the noise level object is already initialized and only the represented power level needs to be adjusted,
the shift operator can be used as a shortcut.

.. literalinclude:: ../../scripts/examples/simulation_noise.py
   :language: python
   :linenos:
   :lines: 19-20

.. autoclass:: hermespy.simulation.noise.model.NoiseModel

.. autoclass:: hermespy.simulation.noise.level.NoiseLevel

.. autoclass:: hermespy.simulation.noise.model.NoiseRealization

.. autoclass:: hermespy.simulation.noise.model.NRT

.. toctree::
   :hidden:

   noise.model.AWGN
   noise.model.AWGNRealization
   noise.level.N0
   noise.level.SNR
   modem.noise.EBN0
   modem.noise.ESN0
   modem.noise.CommunicationNoiseLevel

.. footbibliography::
