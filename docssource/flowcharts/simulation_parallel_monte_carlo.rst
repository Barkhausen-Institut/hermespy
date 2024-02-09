.. mermaid::
   :align: center

   %%{init: {"flowchart":{"useMaxWidth": false}}}%%
   flowchart TD

   pla[Simulated Physical Layer]
   plb[Simulated Physical Layer]

   c{Simulation}

   plc[Simulated Physical Layer]

   pla --> |Evaluations| c 
   c --> |Parameters| pla
   plb --> |Evaluations| c 
   c --> |Parameters| plb
   plc --> |Evaluations| c 
   c --> |Parameters| plc