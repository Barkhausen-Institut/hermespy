function val_clst = clst_sum( val_full, num_subpath )
%CLST_EXTRACT Average value for each sub-path 
% 
% Input:
%   val_full        Data for each cluster [ N x  sum(num_subpath) x O* ]
%   num_subpath     Number of subpaths per cluster [ 1 x L ]
%   weight          An optional weight vector [ N x sum(num_subpath) ]
%
% Output:
%   val_clst        Average for each subpath [ N x L x O ]
%
% M is the number od subpaths for the selected cluster
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

[N,~,M1,M2,M3] = size( val_full );
L = numel( num_subpath );

val_clst = zeros(N,L,M1,M2,M3,class(val_full));

ls = 1;
for l = 1:L
    le = ls + num_subpath(l) - 1;
    
    if le ~= ls
        val_clst(:,l,:) = sum( val_full(:,ls:le,:) , 2 );
    else
        val_clst(:,l,:) = val_full(:,ls,:);
    end

    ls = le + 1;
end

end
