********
HermesPy
********

Welcome to the official documentation of HermesPy, the **Heterogeneous Radio Mobile Simulator**.

HermesPy is a semi-static link-level simulator based on time-driven mechanisms.
It aims to enable the simulation and evaluation of transmission protocols deployed by
multi-RAT wireless electromagnetic communication and sensing devices.
It specifically targets researchers, engineers and students interested in wireless communication and sensing.

.. raw:: html

   <video poster="https://www.barkhauseninstitut.org/fileadmin//user_upload/Filme/2020-12-09-release-trailer.jpg" controls="" no-cookie="" width="100%">
    <source src="https://www.barkhauseninstitut.org/fileadmin/user_upload/Filme/2020-12-09-release-trailer.mp4" type="video/mp4">
   </video>
   <br /> <br />

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

.. toctree::
   :hidden:
   :maxdepth: 3
   :glob:

   self
   features
   installation
   getting_started
   api/api
   references
   developer_hints