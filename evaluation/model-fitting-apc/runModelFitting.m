% Loop through every scenario (instance of model-fitting between
% pre-scanned object model and segmentation of scene point cloud), and do
% RANSAC-based geometric registration to predict object pose
%
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Configuration options (change me)
dataPath = '../../data/apc';

scenarioList = dir(fullfile(dataPath,'scenarios/00*'));
for scenarioIdx = 1:length(scenarioList)
    clusterCallback(scenarioIdx);
end













