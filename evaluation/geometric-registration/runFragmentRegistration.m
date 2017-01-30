% Script to run RANSAC over keypoints and descriptors to predict rigid
% transformations for the geometric registration benchmark 
% 
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Location of scene files (change this and the parameters in clusterCallback.m)
scenePath = '../../data/fragments/7-scenes-redkitchen';
sceneDir = dir(fullfile(scenePath,'*.ply'));
numFragments = length(sceneDir);

% Get fragment pairs
fragmentPairs = {};
fragmentPairIdx = 1;
for fragment1Idx = 1:numFragments
    for fragment2Idx = (fragment1Idx+1):numFragments
        fragment1Name = sprintf('cloud_bin_%d',fragment1Idx-1);
        fragment2Name = sprintf('cloud_bin_%d',fragment2Idx-1);
        fragmentPairs{fragmentPairIdx,1} = fragment1Name;
        fragmentPairs{fragmentPairIdx,2} = fragment2Name;
        fragmentPairIdx = fragmentPairIdx + 1;
    end
end

% Run registration for all fragment pairs
for fragmentPairIdx = 1:size(fragmentPairs,1)
    clusterCallback(fragmentPairIdx);
end