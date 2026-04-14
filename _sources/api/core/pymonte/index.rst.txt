=========
PyMonte
=========

PyMonte is a stand-alone core module of HermesPy,
enabling efficient and flexible Monte Carlo simulations over arbitrary configuration parameter combinations.
By wrapping the core of the `Ray`_ project,
any object serializable by the `pickle`_ standard module can become a system model for a Monte Carlo style simulation
campaign.

.. mermaid::

   flowchart LR

   subgraph gridsection[Grid Section]

      parameter_a(Parameter)
      parameter_b(Parameter)
   end

   object((Investigated Object))
   evaluator_a{{Evaluator}}
   evaluator_b{{Evaluator}}
   evaluator_c{{Evaluator}}

   subgraph sample[Sample]

       artifact_a[(Artifact)]
       artifact_b[(Artifact)]
       artifact_c[(Artifact)]
   end

   parameter_a --> object
   parameter_b --> object
   object ---> evaluator_a ---> artifact_a
   object ---> evaluator_b ---> artifact_b
   object ---> evaluator_c ---> artifact_c

Monte Carlo simulations usually sweep over multiple combinations of multiple parameters settings,
configuring the underlying system model and generating simulation samples from independent realizations
of the model state.
PyMonte refers to a single parameter combination as a grid section,
with the set of all parameter combinations making up the simulation grid.
Each settable property of the investigated object is treated as a potential simulation parameter within the grid,
i.e. each settable property can be represented by an axis within the multidimensional simulation grid.

:class:`.Evaluator` instances extract performance indicators from each investigated object realization, referred to as :class:`.Artifact`.
A set of artifacts drawn from the same investigated object realization make up a single :class:`.MonteCarloSample`.
During the execution of PyMonte simulations between :math:`M_\mathrm{min}` and :math:`M_\mathrm{max}`
are generated from investigated object realizations for each grid section.
The sample generation for each grid section may be aborted prematurely if all evaluators have reached a configured
confidence threshold
Refer to :footcite:t:`2014:bayer` for a detailed description of the implemented algorithm.

.. mermaid::

   flowchart LR

   controller{Simulation Controller}

   gridsection_a[Grid Section]
   gridsection_b[Grid Section]

   sample_a[Sample]
   sample_b[Sample]

   subgraph actor_a[Actor #1]

       object_a((Investigated Object))
   end

   subgraph actor_b[Actor #N]

       object_b((Investigated Object))
   end

   controller --> gridsection_a --> actor_a --> sample_a
   controller --> gridsection_b --> actor_b --> sample_b


The actual simulation workload distribution is visualized in the previous flowchart.
Using `Ray`_, PyMonte spawns a number of :class:`.MonteCarloActor` containers,
with the number of actors depending on the available resources (i.e. number of CPU cores) detected.
A central simulation controller schedules the workload by assigning grid section index tuples as tasks to the actors, which return the resulting simulation Samples after the simulation iteration is completed.

.. _Ray: https://www.ray.io/
.. _pickle: https://docs.python.org/3/library/pickle.html

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 1

   *

.. footbibliography::
