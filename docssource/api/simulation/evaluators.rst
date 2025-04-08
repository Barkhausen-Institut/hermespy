===========
Evaluators
===========

A few evaluators are only available within simulation scenarios,
since they rely on information based on models.

.. list-table::
   :header-rows: 1

   * - Evaluator
     - Description

   * - :class:`SI<hermespy.simulation.evaluators.interference.SI>`
     - Evaluates the self-interference power of a simulated device.

   * - :class:`SSINR<hermespy.simulation.evaluators.interference.SSINR>`
     - Evaluates the signal-to-self-interference-plus-noise ratio of a simulated device.

   * - :class:`SidelobeEvaluator<hermespy.simulation.evaluators.beamforming.SidelobeEvaluator>`
     - Evaluates the directivity of a multi-antenna device.

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 1

   evaluators.*

.. footbibliography::
