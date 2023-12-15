********
HermesPy
********

Welcome to the official documentation of HermesPy, the **Heterogeneous Radio Mobile Simulator**.

HermesPy is a semi-static link-level simulator based on time-driven mechanisms.
It aims to enable the simulation and evaluation of transmission protocols deployed by
multi-RAT wireless electromagnetic devices for both communication and sensing.
It specifically targets researchers, engineers and students interested in wireless communication and sensing.

Users of HermesPy may formulate their own wireless research questions and investigate

   * Spatially distributed heterogeneous communication in up-, down- and side-link
   * Spatially distributed heterogeneous sensing
   * Interference between multi-antenna heterogeneous devices
   * Communication performance indicators such as bit, block and frame error rates and throughput
   * Sensing performance indicators such as detection probabilities, false alarm rates and estimation errors

in both Monte-Carlo style simulations deployable on high-performance computing clusters and in software-defined radio hardware testbeds.
The framework can be operated by means of a compact YAML configuration files or via the native Python :doc:`API<api/api>` and its experimental
:doc:`Matlab Interface<matlab>`, enabling the evaluation of complex scenarios and integration with thrid-party applications with minimal effort.

To get started, visit the :doc:`Getting Started<getting_started>` section to become accustomed with the basic concepts of HermesPy.
For an in-depth view of how to make complex :doc:`API<api/api>` calls and implement your own algorithms within the signal processing
pipelines, visit the :doc:`Tutorials<tutorials>` section.
As a starting point for your own configuration files, visit the :doc:`Examples<examples/examples>` section.
A complete overview of available features can be found in the :doc:`Features<features>` section.

The project is completely open-source and published under the :doc:`GNU AGPL License<license>` on `GitHub`_.
Please cite :footcite:t:`2022:adler` for any results obtained with the help of HermesPy.
Contributions are highly welcome and can be made by means of `GitHub pull requests`_.
For issue reports and feature requests please open a new `GitHub issue`_
or directly contact the current `maintainer`_.

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:

   self
   installation
   getting_started
   tutorials
   api/api
   matlab
   references
   examples/examples
   developer_hints
   license

.. footbibliography::


.. _GitHub: https://github.com/Barkhausen-Institut/hermespy
.. _GitHub pull requests: https://github.com/Barkhausen-Institut/hermespy/pulls
.. _GitHub issue: https://github.com/Barkhausen-Institut/hermespy/issues
.. _maintainer: https://www.linkedin.com/in/jan-adler/
