=================
Core Evaluators
=================

.. inheritance-diagram:: hermespy.core.evaluators.PowerEvaluation hermespy.core.evaluators.PowerArtifact
   :parts: 1

Core evaluators collect basic information about signals at devices.
The following implementations are available:

.. list-table:: Core Evaluators
   :header-rows: 1

   * - Class
     - Description
   * - :py:class:`~hermespy.core.evaluators.ReceivePowerEvaluator`
     - Evaluates the power of a signal impinging onto a device.
   * - :py:class:`~hermespy.core.evaluators.TransmitPowerEvaluator`
     - Evaluates the power of a signl emerging from a device.
   * - :py:class:`~hermespy.core.evaluators.PAPR`
     - Evaluates the Peak-to-Average Power Ratio (PAPR) of a signal.
   * - :py:class:`~hermespy.core.evaluators.SignalExtractor`
     - Extract full sample streams during runtime.

.. autoclass:: hermespy.core.evaluators.PowerEvaluation

.. autoclass:: hermespy.core.evaluators.PowerArtifact

.. toctree::
   :hidden:
   :glob:

   evaluators.*

.. footbibliography::
