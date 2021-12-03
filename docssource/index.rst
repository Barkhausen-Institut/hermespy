**********************
HermesPy Documentation
**********************

Welcome to the official documentation of HermesPy, the **Heterogeneous Radio Mobile Simulator**.

HermesPy is a semi-static link-level simulator based on time-driven mechanisms.
It aims to enable the simulation and evaluation of transmission protocols deployed by
multi-RAT wireless electromagnetic communication and sensing devices.
It specifically targets researchers, engineers and students interested in wireless communication and sensing.

Within an easily expandable, holistic framework users can investigate

* Bit error detection and correction codes
* Communication symbol mappings
* Signal modulations and their respective waveforms
* Wireless communication channels
* Channel pre-coding and equalization
* Interference between multiple multi-antenna devices

HermesPy may be used in one of two modes of operation:

#. As a command line tool for Monte-Carlo simulations of wireless transmission scenarios.
   See :doc:`/getting_started_command_line` for further information.
#. As a plug-in library for the simulation of realistic wireless transmissions within your own python projects.
   See :doc:`/getting_started_library` for further information.

We highly recommend studying the instructions in the respective getting-started sections prior to working with Hermes.
In order to install HermesPy on your system, follow the instructions outlined in
:doc:`/installation`.
Feature set, and release plan can be found on `GitHub <https://github.com/Barkhausen-Institut/hermespy/>`_.

.. toctree::
   :hidden:
   :maxdepth: 3
   :glob:

   self
   installation
   getting_started
   api/modules
