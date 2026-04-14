.. mermaid::

   flowchart TD
      sim[Simulation];
      loop[Hardware Loop];
      beam[Beamforming];
      channel[Channel];
      core[Core];
      fec[FEC];
      jcas[JCAS];
      tools[Tools];
      modem[Modem];
      radar[Radar];
      tools[Tools];

      sim --> core;
      loop --> core;
      beam --> core;
      sim --> channel;
      modem --> fec;
      modem --> core;
      jcas --> modem;
      jcas --> radar;
      radar --> core;
      radar --> beam;

      click beam href "/api/beamforming/index.html";
      click channel href "/api/channel/index.html";
      click core href "/api/core/index.html";
      click fec href "/api/fec/index.html";
      click loop href "/api/hardware_loop/index.html";
      click jcas href "/api/jcas/index.html";
      click modem href "/api/modem/index.html";
      click radar href "/api/radar/index.html";
      click sim href "/api/simulation/index.html";
      click tools href "/api/tools/index.html";