In this example we simulate a frame of QAM modulated overlapping chirps.

A bandwidth of 100 MHz is considered with chirps separated at the Nyquist
sampling rate. Modulation is 16-QAM, yielding an uncoded bit rate of 400 Mbps.
Chirps have a duration of 1 \mu s.
The interchirp interference is compensated by an MMSE block equalizer.

A non-linear amplifier following Rapp's model is also considered.

Each frame transmits 10 unmodulated non-overlapping chirps at the beginning,
followed by 1000 modulated chirps.

Channel is AWGN.

