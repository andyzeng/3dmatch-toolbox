% Script to run create evaluation .log files from predicted rigid
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

descriptorName = '3dmatch';
dataPath = '../../data/fragments'; % Location of scene files
intermPath = '../../data/fragments/intermediate-files'; % Location of intermediate files holding keypoints and descriptor vectors
savePath = '../../data/fragments/intermediate-files'; % Location to save evaluation .log file
sceneList = {'iclnuim-livingroom1','iclnuim-livingroom2','iclnuim-office1','iclnuim-office2'};

totalRecall = []; totalPrecision = [];
for sceneIdx = 1:length(sceneList)
    
    % List fragment files
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    sceneDir = dir(fullfile(scenePath,'*.ply'));
    numFragments = length(sceneDir);
    fprintf('%s\n',scenePath);

    % Loop through registration results and write a log file
    logPath = fullfile(savePath,sceneList{sceneIdx},sprintf('%s.log',descriptorName));
    fid = fopen(logPath,'w');
    for fragment1Idx = 1:numFragments
        for fragment2Idx = (fragment1Idx+1):numFragments
            fragment1Name = sprintf('cloud_bin_%d',fragment1Idx-1);
            fragment2Name = sprintf('cloud_bin_%d',fragment2Idx-1);

            resultPath = fullfile(intermPath,sceneList{sceneIdx},sprintf('%s-registration-results',descriptorName),sprintf('%s-%s.rt.txt',fragment1Name,fragment2Name));

            resultRt = dlmread(resultPath,'\t',[2,0,5,3]);
            resultNumInliers = dlmread(resultPath,'\t',[1,0,1,0]);
            resultInlierRatio = dlmread(resultPath,'\t',[1,1,1,1]);
            resultAlignRatio = dlmread(resultPath,'\t',[1,2,1,3]);
            
            % Check if surface overlap is above some threshold
            if resultAlignRatio(1) > 0.23
                fprintf(fid,'%d\t %d\t %d\t\n',fragment1Idx-1,fragment2Idx-1,numFragments);
                fprintf(fid,'%.10f\t%.10f\t%.10f\t%.10f\n',resultRt');
            end
        end
    end
    fclose(fid);
    
end
