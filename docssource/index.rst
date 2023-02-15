********
HermesPy
********

Welcome to the official documentation of HermesPy, the **Heterogeneous Radio Mobile Simulator**.

HermesPy is a semi-static link-level simulator based on time-driven mechanisms.
It aims to enable the simulation and evaluation of transmission protocols deployed by
multi-RAT wireless electromagnetic communication and sensing devices.
It specifically targets researchers, engineers and students interested in wireless communication and sensing.

Please cite :footcite:t:`2022:adler` for any results obtained with the help of HermesPy.
For issue reports, feature requests or contributions please open a `GitHub issue`_
or directly contact the current `maintainer`_.

.. raw:: html

   <video poster="https://www.barkhauseninstitut.org/fileadmin//user_upload/Filme/2020-12-09-release-trailer.jpg" controls="" no-cookie="" width="100%">
    <source src="https://www.barkhauseninstitut.org/fileadmin/user_upload/Filme/2020-12-09-release-trailer.mp4" type="video/mp4">
   </video>
   <br /> <br />

Within an easily expandable, holistic framework users can investigate

* Bit error detection and error-correcting codes
* Communication symbol mappings
* Signal modulations and their respective waveforms
* Wireless communication channels
* Channel pre-coding and equalization
* Interference between multiple multi-antenna heterogeneous devices
* Sensing algorithms and KPIs

in both simulations and hardware testbeds.

.. carousel::
   :show_controls:
   :show_indicators:
   :show_shadows:

   .. figure:: images/getting_started_ber_evaluation.png

      Communication KPIs such as BER, BLER, FER and throughput

   .. figure:: images/getting_started_constellation_low_noise.png

      Communication symbol mappings

   .. figure:: images/getting_started_signal_plot_preamble.png

      Signal modulations and their respective waveforms

   .. figure:: images/eye.png

      Signal modulations and their eye diagrams

   .. figure:: images/index_beamforming.png

      Beamforming

   .. figure:: images/index_radar.png

      Sensing KPIs

   .. figure:: images/getting_started_simulation_multidim_drx.png

      Multidimensional parameter sweeps


HermesPy may be used in one of two modes of operation:

#. As a command line tool for Monte-Carlo evaluations of wireless transmissions.
   This enables the definition of complex scenarios by means of compact YAML configuration files.
   See :ref:`Getting Started -> Command Line Tool<GettingStarted_CommandLineTool>` for further information.
#. As a plug-in library for the evaluation of realistic wireless scenarios within your own python projects.
   Due to its modular design, parts of the simulation chain can be operated stand-alone to be customized for individual requirements,
   or the whole framework can be operated by its native Python interface.
   See :ref:`Getting Started -> Library<GettingStarted_Library>` for further information.

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:

   self
   features
   installation
   getting_started
   tutorials
   examples/examples
   api/api
   matlab
   references
   developer_hints

.. footbibliography::


.. _GitHub issue: https://github.com/Barkhausen-Institut/hermespy/issues
.. _maintainer: https://www.linkedin.com/in/jan-adler/
