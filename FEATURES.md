# Features

The current releases encompasses the following:
## Platform  Release (December 2020)

* **Simulation**
  * Framework for definition of a multi-RAT link with multiple modems through parameter files
  * main simulation loop and statistics
  * BER/BLER statistics
  * confidence interval is considered as a stopping criterion
  * theoretical results (depending on simulation scenario)
  * power spectral density and time-frequency analysis
  * support for multiple transmitter and receivers

* **Channel**
  * multipath
  * (SISO) macrocell COST-259 and exponential channel models
  * stochastic multipath channel (with arbitrary power delay profile, Rayleigh/Rice fading and antenna correlation matrix)
  * Quadriga channel model

* **Modem**
  * generic BPSK modem with AWGN channel and root-raised cosine filter
  * chirp FSK modem
  * generic OFDM modem
  * higher-order PSK/QAM modem with soft-output detector
  * sampling rate adaptation

- **Codes**
  - abstract base class, RepetitionEncoder as example encoder
  
## Release Plan

We provide future releases on a half-year basis starting in April 2021.
### April 2021 / "5G Ready"

- **Simulation**
  - Open-Loop MIMO support (spatial multiplexing, space-time codes)
  - improving extensibility of platform approach
- **Channel**
  - 5G PHY channel model
- **Modem**
  - 5G Frame including reference symbols
- **Codes**
  - LDPC codes
- **Other**
  - User manual
  - bugfixes
  - CI

### October 2021 / "5G++"

- **Simulation**
- **Channel**
  - IQ imbalance
  - phase noise
- **Modem**
  - Non-linear amplifiers
  - equalization for SC modulation
  - channel estimation
  - flexible GFDM modem
  - beamforming and antenna patterns
- **Codes**
- **Other**