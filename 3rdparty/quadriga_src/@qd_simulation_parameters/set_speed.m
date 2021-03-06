function set_speed( qd_simulation_parameters, speed_kmh, sampling_rate_s )
%SET_SPEED This method can be used to automatically calculate the sample density for a given mobile speed
%
% Calling object:
%   Single object
%
% Input:
%   speed_kmh
%   speed in [km/h]
%
%   sampling_rate_s
%   channel update rate in [s]
%
% 
% QuaDRiGa Copyright (C) 2011-2019
% Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
% Fraunhofer Heinrich Hertz Institute, Einsteinufer 37, 10587 Berlin, Germany
% All rights reserved.
%
% e-mail: quadriga@hhi.fraunhofer.de
%
% This file is part of QuaDRiGa.
%
% The Quadriga software is provided by Fraunhofer on behalf of the copyright holders and
% contributors "AS IS" and WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, including but not limited to
% the implied warranties of merchantability and fitness for a particular purpose.
%
% You can redistribute it and/or modify QuaDRiGa under the terms of the Software License for 
% The QuaDRiGa Channel Model. You should have received a copy of the Software License for The
% QuaDRiGa Channel Model along with QuaDRiGa. If not, see <http://quadriga-channel-model.de/>. 


qd_simulation_parameters.samples_per_meter = 1/( speed_kmh/3.6 * sampling_rate_s);

end

