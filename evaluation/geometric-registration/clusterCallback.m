% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function clusterCallback(jobID)

    % Configuration options (change me)
    descriptorName = '3dmatch';
    scenePath = '../../data/fragments/7-scenes-redkitchen'; % Location of scene fragment point clouds
    intermPath = '../../data/fragments/intermediate-files/7-scenes-redkitchen'; % Location of intermediate files holding keypoints and descriptor vectors
    
    % Add libraries
    addpath(genpath('../../core/external'));

    % List out scene fragments Location of scene files
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
    
    if jobID > size(fragmentPairs,1)
        return;
    end
    
    fragment1Name = fragmentPairs{jobID,1};
    fragment2Name = fragmentPairs{jobID,2};
    fprintf('Registering %s and %s: ',fragment1Name,fragment2Name);

    % Get results file
    resultPath = fullfile(intermPath,sprintf('%s-registration-results',descriptorName),sprintf('%s-%s.rt.txt',fragment1Name,fragment2Name));
    if exist(resultPath,'file')
        fprintf('\n');
        return;
    end

    % Compute rigid transformation that aligns fragment 2 to fragment 1
    [estimateRt,numInliers,inlierRatio,ratioAligned] = register2Fragments(scenePath,intermPath,fragment1Name,fragment2Name,descriptorName);
    fprintf('%d %f %f %f\n',numInliers,inlierRatio,ratioAligned(1),ratioAligned(2));

    % Save rigid transformation
    fid = fopen(resultPath,'w');
    fprintf(fid,'%s\t %s\t\n%d\t %15.8e\t %15.8e\t %15.8e\t\n',fragment1Name,fragment2Name,numInliers,inlierRatio,ratioAligned(1),ratioAligned(2));
    fprintf(fid,'%15.8e\t %15.8e\t %15.8e\t %15.8e\t\n',estimateRt');
    fclose(fid);
end