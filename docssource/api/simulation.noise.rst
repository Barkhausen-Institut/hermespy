======
Noise
======

Hardware noise is one of the primary factors limiting the performance of wireless systems in both communication and sensing applications.
While any processing step introduces noise to some extent, Hermes assumes receiving front-ends to be the primary source of noise power added to any incoming signal.

HermesPy's noise model consists of a tandem of two components, the :class:`Noise<hermespy.simulation.noise.noise.Noise>` model
and its :class:`NoiseRealization<hermespy.simulation.noise.noise.NoiseRealization>`.

.. mermaid::
   :align: center

   flowchart LR

   model[Noise] -->|realize| realization[NoiseRealization]

   click model "simulation.noise.Noise.html" "Noise"
   click realization "simulation.noise.NoiseRealization.html"

The currently available implementations of noise models are

.. list-table::
   :header-rows: 1

   * - Model
     - Realization
     - Description

   * - :doc:`simulation.noise.AWGN`
     - :doc:`simulation.noise.AWGNRealization`
     - Complex additive white Gaussian noise.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>` thermal
noise model is as simple as setting its :attr:`noise<hermespy.simulation.simulated_device.SimulatedDevice.noise>` porperty to an instance of a noise model.

.. literalinclude:: ../scripts/examples/simulation_noise.py
   :language: python
   :linenos:
   :lines: 6-11

Of course, the abstract *Noise* model in the above snippet has to be replaced with a specific implementation
from the above table.
The actual noise power can be adjusted indepentely from the model by either setting the default power property or
specyfing the device's signal to nois ratio (SNR) property.

.. literalinclude:: ../scripts/examples/simulation_noise.py
   :language: python
   :linenos:
   :lines: 13-18

.. toctree::
   :hidden:

   simulation.noise.AWGN
   simulation.noise.AWGNRealization
   simulation.noise.Noise
   simulation.noise.NoiseRealization

.. footbibliography::
