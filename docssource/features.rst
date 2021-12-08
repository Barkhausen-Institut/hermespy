*****
About
*****

Features
========

The current feature-set of HermesPy is steadily expanding and includes
(new features in **bold**)

Modulation and Coding
---------------------

* Generic PSK/QAM/PAM modem with square pulses, (root)-raised-cosine filters or FMCW
* Linear equalizers for non-orthogonal FMCW pulses in AWGN
* LLR calculation for BPSK/QAM/16-/64-/256-QAM
* Chirp FSK 
* Generic OFDM with arbitrary allocation of data and reference symbols in each resource element
* DFT-spread supported
* **Extended GFDM framework**
* Repetition and LDPC codes
* **Faster LDPC decoder**
* **3GPP-like scrambler**
* **Block interleaver**
* **CRC overhead**
* Transmit diversity (Alamouti) with 2 or 4 tx antennas
* Open-loop spatial multiplexing with linear receivers
* Receive diversity (SC or MRC)
* **Radar detection for FMCW**
* **Channel estimation for OFDM**

Channel and Interference Model
------------------------------

* Time-variant multipath channel with arbitrary power delay profile, Rice/Rayleigh fading
* COST-259 macrocell model
* 5G TDL model
* MIMO support with antenna correlation, following Kronecker model
* Interface to Quadriga channel model (requires Matlab or Octave)
* Interference among different modems, with arbitrary transmit powers for different transmitters
* Support for transmitters using different carrier frequencies and bandwidths
* **Single-target** radar channel model

RF Chain
--------

* Memoryless non linear power amplifier following an ideal clipper, Rapp's, Saleh's or any arbitrary AM/AM AM/PM responses
* **Random time offset**
* **I/Q imbalance**

Simulation
----------

* **Installation as a Python library**
* **Greater modularity and standalone usage of simulator classes**
* Full configuration using **YAML** settings files
* Script for plotting different simulations in one graph
* Drops containing several frames
* Support for multiple transmitter and receivers
* User manual provided

Statistics
----------

* BER/FER statistics
* Confidence interval is calculated and may be considered as a stopping criterion
* Theoretical results available (depending on simulation scenario)
* Power spectral density and time-frequency analysis
* **Time-domain waveform plots**
* **Constellation plots**

Known Limitations
=================

The known limitations currently include

* No native simulation parallelization


Release Plan
============

Full releases with a new set of features will be provided on a half-yearly basis, with software patches in between.
For the next release, the current plan is

* **Modulation and coding**

   * Radar Detection for FMCW and OFDM
   * Beamforming

* **Channel and interference model**

   * Hardware in the Loop

* **RF chain**

   * Phase Noise