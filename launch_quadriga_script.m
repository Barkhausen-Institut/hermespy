%Add the path to Quadriga source code folder
addpath(path_quadriga_src); %('C:\Marco_Code\Quadriga API\quadriga_src'); %<<<Python 

function set_seed(seed)
  v = version;
  if ~isempty(strfind(v, 'R20')) % matlab
    RandStream.setGlobalStream(RandStream('mt1937ar', 'seed', seed));
  else 
    rand('seed', seed);
  endif
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input parameters:

% samples per second (at least double of bandwidth)
samp_rate = sampling_rate; %[1e6, 10e6]; % Msps, for each Tx can be different, %<<<Python
% second per tap/path, along time delay axis
samp_time = 1./samp_rate;

sec_per_snap = 10e-3; % named 'update_rate' in documentation
snap_per_sec = 1./sec_per_snap;% named 'fT' in documentation (pag.86)
% moreover: 'sample_density' is snaps per half-wavelength,
%           'sample_per_meter' is snaps per meter

% carrier frequencies
carrier_frequencies = carriers %[1e9,3e9]; %Hz, %<<<Python

numb_tx = number_tx; %2; %<<<Python
ant_number_tx = txs_number_antenna; %<<<Python
ant_kind_tx = tx_antenna_kind; %'lhcp-rhcp-dipole'; %see documentation pag. 32, %<<<Python
% try 'half-wave-dipole' for one port antenna.
% try 'lhcp-rhcp-dipole' for two ports antennas.

ant_rot_flag_tx = 0;
ant_rot_angle_tx = 90;
ant_rot_axis_tx = 'x';

pos_tx = transpose(tx_position); %[[0;0;20],[0;0;15]]; %<<<Python
%                x;y;z
% base station heights:
%   below-roof-top = 3 - 6 m
%   UMA macro = 20 - 25 m
%   UMI micro = 10 - 15 m
%   indoor = 1 - 6 m

numb_rx = number_rx; %3; %number of receivers, %<<<Python
ant_number_rx = rxs_number_antenna; %<<<Python
ant_kind_rx = rx_antenna_kind; % cf. tx antenna; %see documentation pag. 32, %<<<Python

ant_rot_flag_rx = 0;
ant_rot_angle_rx = 90;
ant_rot_axis_rx = 'x';

 %starting point of the receiver  in m
pos_rx = transpose(rx_position); %[[100;100;2],[-50;-50;2],[50;100;2]]; %<<<Python
track_length_rx = tracks_length; %[100,60,150]; %linear track length in m, %<<<Python
track_angle_rx = tracks_angle; %[0,45,90]; %linear track direction angle, degrees, %<<<Python
track_speed_rx = tracks_speed; %[0,0,0]; %meter per second, %<<<Python
% Examples: 
%   walk: 0.8 - 1.5 m/s
%   car: 14 - 30 m/s
%   tram: 5 - 20 m/s
%   bike: 3 - 8 m/s

%label of the scenario -> look in quadriga_src/config
scenario_label = scenario_label; %'3GPP_38.901_UMa_LOS'; %<<<Python

%Spatial theorem must be fulfilled
if ( snap_per_sec < 4 * max(track_speed_rx) / min(3e8./carrier_frequencies) )   
    warning('snap_per_sec not enough!');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setting and init.

set_seed(seed); %<<<Python
quadriga_settings = qd_simulation_parameters; 
quadriga_settings.center_frequency = carrier_frequencies; 
quadriga_settings.use_absolute_delays = 0; % LOS path comes at 0 sec. 
quadriga_settings.use_random_initial_phase = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create antenna
txs_number_antenna
ant_tx = qd_arrayant(ant_kind_tx);
ant_tx.visualize;

[ gain_dBi_tx, pow_max_tx ] = ant_tx.calc_gain;

if (ant_rot_flag_tx == 1)
    ant_tx.rotate_pattern( ant_rot_angle_tx, ant_rot_axis_tx );
    %                         degreee,          axis string
    %by default antenna pointing is on X axis, towards East

    ant_ports_tx = ant_tx.no_elements;   
end

ant_rx = qd_arrayant(ant_kind_rx);
ant_rx.visualize;
[ gain_dBi_rx, pow_max_rx ] = ant_rx.calc_gain;

if (ant_rot_flag_rx == 1)
    ant_rx.rotate_pattern( ant_rot_angle_rx, ant_rot_axis_rx );
    %                         degreee,          axis string
    %by default antenna pointing is on X axis, towards East

    ant_ports_rx = ant_rx.no_elements;    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Layout

layout = qd_layout(quadriga_settings);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Transmitters

layout.no_tx = numb_tx;

%associate antennas, positions and names
for itx = 1:layout.no_tx
    layout.tx_name(itx) = {['Tx',num2str(itx-1)]};
    
    ant_tx.no_elements = txs_number_antenna(itx);
    layout.tx_array(itx) = copy(ant_tx);
    layout.tx_position(:,itx) = pos_tx(:, itx) ;   
end

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Receivers

layout.no_rx = numb_rx;

%associate antennas, positions and names
for irx = 1:layout.no_rx
    ant_rx.no_elements = rxs_number_antenna(irx);
    
    layout.rx_name(irx) = {['Rx',num2str(irx-1)]};
    layout.rx_array(irx) = copy(ant_rx); 
    
    if ( track_speed_rx(irx) > 0  ) %mobile case
        track_rx = qd_track.generate('linear', track_length_rx(irx), deg2rad(track_angle_rx(irx)));       
        track_rx.name = ['track',num2str(irx-1)];
        
        track_rx.initial_position =  pos_rx(:, irx); 
        track_rx.set_speed(track_speed_rx(irx));
        
        track_rx.scenario = {scenario_label};  
        
        % computing antenna orientation along track
        calc_orientation(track_rx);
        
        layout.rx_track(irx) = track_rx;

    else %static case
        
        track_rx = qd_track.generate('linear', 1, 0); 
        track_rx.name = ['track',num2str(irx-1)];
        
        track_rx.initial_position =  pos_rx(:, irx); 
        track_rx.no_snapshots = 1;
        
        track_rx.scenario = {scenario_label};  
        
        calc_orientation(track_rx);
        
        layout.rx_track(irx) = track_rx; 

    end
end

layout.visualize();  
view(0,90);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Channels generation

if ( all(track_speed_rx)  ) %mobile case   
    %Generating a layout.channel impulse responses
    [channels, builder] = layout.get_channels(sec_per_snap, 1 );
else    
    %Generating a layout.channel impulse responses
    [channels, builder] = layout.get_channels();
end

%%%%%%%%%%%%%%% results format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c(irx,itx,iff).coeff(iarx,iatx,ipath,isnap);
%
%   /irx is rx index
%   /itx is tx index
%   /iff is carrier freq. index   
%       /iarx is rx antenna elem index  
%       /iatx is tx antenna elem index 
%       /ipath is path index (Taps)
%       /isnap is snapshot index (Track points)  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%creating the absolute time axis
numb_snap = channels(1,1,1).no_snap;
%sec, in line with 'sec_per_snap'
time_axis = [0: sec_per_snap: (sec_per_snap)*(numb_snap-1)]; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Paths re-sampling

%interpolating along the excess delays with a constant sampling step
%there might be zeros 'tails' in the coeff/delay vectors at the end 

%Outputs:
cirs = []; % size ~ (irx, itx).[iarx,iatx,ipath,isnap]

%packing the results for each Rx, keeping each channel separated,
%the interference are meant to be computed in Hermes...

for irx = 1:layout.no_rx    
    for itx = 1:layout.no_tx
        %assuming that each Tx has a distinct carrier frequency assigned to itself
        ifrq = itx;       
        if numb_snap > 1
            %re-sampling, each Tx can have different bandwidth/sampling rate
            channels(irx,itx,ifrq) = quantize_delays( channels(irx,itx,ifrq), samp_time(itx), 1 );
        endif  
        cirs(irx, itx).path_impulse_responses = channels(irx, itx, ifrq).coeff;
        cirs(irx, itx).tau = channels(irx, itx, ifrq).delay;
    end
end