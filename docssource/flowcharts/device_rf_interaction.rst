.. mermaid::
   :align: center

   graph LR
   
    subgraph rftx [RF Tx Chain]
        direction LR
        dac[DAC] --> iqtx[I/Q] --> pntx[PN] --> pa[PA]
    end

    pa --- txsplit(( ))--> mctx[MC] --> anttx[ANT] --> chantx{{Tx}}

    subgraph rfrx [RF Rx Chain]
        direction LR
        lna[LNA] --> pnrx[PN] --> iqrx[I/Q] --> adc[ADC]
    end

    chanrx{{Rx}} --> antrx[ANT] --> mcrx[MC] --- rxmerge((+)) --> lna
    txsplit --> iso[ISO] --- rxmerge
