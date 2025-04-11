=================
Power Evaluators
=================

.. inheritance-diagram:: hermespy.core.evaluators.PowerEvaluation hermespy.core.evaluators.PowerArtifact hermespy.core.evaluators.PowerResult
   :parts: 1

Power evaluators collect basic information about the power of signals at devices.
The following implementations are available:

.. list-table:: Power Evaluators
   :header-rows: 1

   * - Class
     - Description
   * - :py:class:`~hermespy.core.evaluators.ReceivePowerEvaluator`
     - Evaluates the power of a signal impinging onto a device.
   * - :py:class:`~hermespy.core.evaluators.TransmitPowerEvaluator`
     - Evaluates the power of a signl emerging from a device.
   * - :py:class:`~hermespy.core.evaluators.PAPR`
     - Evaluates the Peak-to-Average Power Ratio (PAPR) of a signal.

.. autoclass:: hermespy.core.evaluators.PowerEvaluation

.. autoclass:: hermespy.core.evaluators.PowerArtifact

.. autoclass:: hermespy.core.evaluators.PowerResult

.. toctree::
   :hidden:
   :glob:

   evaluators.*

.. footbibliography::
