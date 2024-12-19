.. mermaid::
   :align: center

   %%{init: {"flowchart":{"useMaxWidth": false, "curve": "linear"}}}%%
   graph LR

   subgraph phys [Simulated Physical Layer]
      direction LR

      dai[SimulatedDevice A]
      dbi[SimulatedDevice B]

      caa{{Channel A,A}}
      cab{{Channel A,B}}
      cba{{Channel B,A}}
      cbb{{Channel B,B}}

      dao[SimulatedDevice A]
      dbo[SimulatedDevice B]
   end

   dai -.-> caa -.-> dao
   dai -.-> cab -.-> dbo
   dbi -.-> cbb -.-> dbo
   dbi -.-> cba -.-> dao

   click dai "/api/simulation.simulated_device.SimulatedDevice.html" "Simulated Device"
   click dbi "/api/simulation.simulated_device.SimulatedDevice.html" "Simulated Device"
   click dao "/api/simulation.simulated_device.SimulatedDevice.html" "Simulated Device"
   click dbo "/api/simulation.simulated_device.SimulatedDevice.html" "Simulated Device"
   click caa "/api/channel/channel.Channel.html" "Channel Model"
   click cab "/api/channel/channel.Channel.html" "Channel Model"
   click cba "/api/channel/channel.Channel.html" "Channel Model"
   click cbb "/api/channel/channel.Channel.html" "Channel Model"
