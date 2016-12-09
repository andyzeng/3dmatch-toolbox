% Script to evaluate .log files for the real-world geometric registration
% benchmark, in the same spirit as Choi et al 2015. Please see:
%
% http://redwood-data.org/indoor/regbasic.html
% https://github.com/qianyizh/ElasticReconstruction/tree/master/Matlab_Toolbox
%
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Locations of evaluation files
sceneList = {'../../data/fragments/iclnuim-livingroom1-evaluation' ...
             '../../data/fragments/iclnuim-livingroom2-evaluation' ...
             '../../data/fragments/iclnuim-office1-evaluation' ...
             '../../data/fragments/iclnuim-office2-evaluation'};
         
% Load Elastic Reconstruction toolbox
addpath(genpath('external'));

% Compute precision and recall
totalRecall = []; totalPrecision = [];
for sceneIdx = 1:length(sceneList)
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    
    % Compute registration error
    gt = mrLoadLog(fullfile(scenePath,'gt.log'));
    gt_info = mrLoadInfo(fullfile(scenePath,'gt.info'));
    result = mrLoadLog(fullfile(scenePath,'3dmatch.log'));
    [recall,precision] = mrEvaluateRegistration(result,gt,gt_info);
    totalRecall = [totalRecall;recall];
    totalPrecision = [totalPrecision;precision];
end
fprintf('Mean registration recall: %f precision: %f\n',mean(totalRecall),mean(totalPrecision));
