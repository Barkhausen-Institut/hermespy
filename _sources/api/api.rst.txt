============
HermesPy API
============

This section describes all parts of the HermesPy Application Programming Interface.
HermesPy consist of the a self-titled namespace featuring multiple sub-modules,
each tackling a different aspect of signal processing for wireless systems.
The subpackages may have interdependencies, but in general, the root of dependencies is provided by the core package:

.. include:: ../flowcharts/module_structure.rst

.. list-table::
   :header-rows: 1

   * - Module
     - Description

   * - :doc:`/api/simulation/index`
     - Simluation of wireless scenarios, numerical models of the physical layer.

   * - :doc:`/api/channel/index`
     - Collection of variious wireless channel models.

   * - :doc:`/api/hardware_loop/index`
     - Interfaces to software-defined radio hardware, compatible with the core architecture.

   * - :doc:`/api/modem/index`
     - Signal processing pipelines for communication applications as commonly found in wireless modems.

   * - :doc:`/api/radar/index`
     - Signal processing pipelines for sensing applications as commonly found in radar systems.

   * - :doc:`/api/fec/index`
     - Forward error correction coding and decoding algorithms. Dependency of the modem module.

   * - :doc:`/api/jcas/index`
     - Joint communication and sensing algorithms, including waveform design and resource allocation.

   * - :doc:`/api/beamforming/index`
     - Beamforming algorithms, both digital and analog.

   * - :doc:`/api/core/index`
     - Core architecture, including serialization, device description and Monte Carlo distribution management.


.. toctree::
   :maxdepth: 1
   :hidden:
   :glob:

   modem/index
   radar/index
   channel/index
   simulation/index
   hardware_loop/index
   core/index
   fec/index
   jcas/index
   beamforming/index
   tools/index