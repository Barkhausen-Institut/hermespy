================
Sidelobe Level
================

.. inheritance-diagram:: hermespy.simulation.evaluators.beamforming.SidelobeEvaluator hermespy.simulation.evaluators.beamforming.SidelobeEvaluation
   :parts: 1

The sidelove evaluator measures the directivity of a given signal pattern
by comparing the power of the assumed main lobe towards a desired direction
with the overall emitted power sampled at a set of candidate directions.
The directivity

.. math:: 

   P_{\mathrm{Dir}} = \frac{P_{\mathrm{Main}}}{P_{\mathrm{Total}}}

approaches :math:`1` when the main lobe is very narrow and the sidelobes are very low and zero when the main lobe is very wide and the sidelobes are very high.

.. literalinclude:: ../../scripts/examples/simulation_evaluation_sidelobe.py
   :language: python
   :linenos:
   :lines: 38-52

.. autoclass:: hermespy.simulation.evaluators.beamforming.SidelobeEvaluator

.. autoclass:: hermespy.simulation.evaluators.beamforming.SidelobeEvaluation
    
.. footbibliography::