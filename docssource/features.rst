*****
About
*****

Features
========

The feature set of HermesPy is steadily expanding and currently includes
(features in latest release are in **bold**)

Modulation and Coding
---------------------

* Coding
    * Repetition Codes
    * 3GPP-like LDPC codes (**faster implementation with C++**)
    * **3GPP-like scrambler**
    * **Block interleaver**
    * **CRC overhead**
* Modulation and Waveforms
    * Generic PSK/QAM/PAM modem with square pulses, (root)-raised-cosine filters or FMCW
    * Chirp FSK (a.k.a. chirp spread spectrum)
    * OFDM frame with arbitrary allocation of data and reference symbols in each resource element
* multiple antennas and precoding
    * Transmit diversity (Alamouti) with 2 or 4 tx antennas
    * Open-loop spatial multiplexing with linear receivers
    * DFT-spread for OFDM
    * **Extended GFDM framework** :cite:p:`2018:nimr`
* Receiver algorithms
    * LLR calculation for BPSK/QAM/16-/64-/256-QAM
    * Linear equalizers for non-orthogonal FMCW pulses in AWGN
    * **Channel estimation for OFDM**
    * Receiver diversity (SC or MRC)
    * **Radar detection for FMCW**

Channel and Interference Model
------------------------------

* Time-variant multipath channel with arbitrary power delay profile, Rice/Rayleigh fading
* COST-259 macrocell model :cite:p:`2004:3GPP:TR25943`
* 5G TDL model :cite:p:`2017:3GPP:TR38901`
* MIMO support with antenna correlation, following Kronecker model
* Interface to `Quadriga <https://quadriga-channel-model.de/>`_ channel model (requires Matlab or Octave)
* Interference among different modems, with arbitrary transmit powers for different transmitters
* Support for transmitters using different carrier frequencies and bandwidths
* **Single-target** radar channel model

RF Chain
--------

* Memoryless non linear power amplifier
    * ideal clipper,
    * Rapp's model :cite:p:`1991:rapp`,
    * Saleh's model :cite:p:`1981:saleh`,
    * arbitrary AM/AM AM/PM responses
* **Random time offset**
* **I/Q imbalance**

Simulation
----------

* **Installation as a Python library**
* **Greater modularity and standalone usage of simulator classes**
* Full configuration using **YAML** settings files
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
* Radar KPIs missing (only delay-Doppler map is generated)

Release Plan
============

Full releases with a new set of features will be provided on a half-yearly basis, with software patches in between.
For the next release in April 2022, the current plan is

* **Modulation and coding**

   * Radar Detection for FMCW and OFDM
   * Beamforming

* **Channel and interference model**

   * Hardware in the Loop
   * 3GPP clustered delay line

* **RF chain**

   * Phase Noise