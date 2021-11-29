=================
Command Line Tool
=================

This section outlines how to use **HermesPy** as a command line tool
and provides some reference examples to get new users accustomed with the process of configuring scenarios.

Once **HermesPy** is installed within any python environment,
users may call the command line interface by entering the command ``hermes``
in both Linux and Windows environments.
Check :doc:`/api/hermespy.bin.hermes` for a detailed description of all command line options.

In short, entering

.. code-block:: console

    hermes -p /path/to/settings -o /path/to/output

is the most common use-case of the command line interface.
All configuration files located under */path/to/settings* are parsed and interpreted as
an executable scenario configuration.
The configuration is subsequently being executed.
All data resulting from this execution will be stored within */path/to/output*.

-----------
First Steps
-----------

Let's start by configuring a basic simulation scenario.
It should consist of:

#. A single transmitting modem, a single receiving modem, both featuring a single ideal antenna
#. A Quadrature-Amplitude modulation scheme
#. A central carrier frequency of 1GHz
#. An ideal channel between both transmitting and receiving modem

This scenario is represented by the following *scenario.yml* file:

.. code-block:: yml
   :linenos:

    !<Scenario>

    Modems:

      - Transmitter

        carrier_frequency: 1e9

        WaveformPskQam:
            num_data_symbols: 100

      - Receiver

        carrier_frequency: 1e9

        WaveformPskQam:
            num_data_symbols: 100


A second *simulation.yml* file determines the execution behaviour of the scenario.
For now, the baseband-waveforms as well as the mapped symbol constellations
at both the transmitter and receiver should be plotted:

.. code-block:: yml
   :linenos:

    !<Simulation>

    snr: [20]

    plot_drop: true
    plot_drop_transmitted_signals: true
    plot_drop_received_signals: true
    plot_drop_transmitted_symbols: true
    plot_drop_received_symbols: true

Assuming both *scenario.yml* and  *simulation.yml* are located within */path/to/settings*, calling

.. code-block:: console

    hermes -p /path/to/settings

will result in the rendering of four plots displaying the respective information.