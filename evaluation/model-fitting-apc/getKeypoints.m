% Randomly sample surface keypoints from the object model and segmentation
% point clouds, and save keypoints to binary files
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
dataPath = '../../data/apc'; % Path to apc data

% Compute keypoints for segmentation point clouds
scenarioDir = dir(fullfile(dataPath,'scenarios/00*'));
for scenarioIdx = 0:(length(scenarioDir)-1)
    scenarioPath = fullfile(dataPath,'scenarios',sprintf('%06d',scenarioIdx));
    fprintf('%s\n',scenarioPath);

    % Get scenario info
    fileID = fopen(fullfile(scenarioPath,'info.txt'),'r');
    sceneName = fscanf(fileID,'scene: %s\n');
    objName = fscanf(fileID,'object: %s\n');
    fclose(fileId);
    
    % Load segmentation point cloud and TDF voxel grid
    segmPointCloud = pcread(fullfile(scenarioPath,'segmentation.ply'));
    matchSegmVoxFile = fullfile(scenarioPath,'segmentation.TDF.bin');
    fileID = fopen(matchSegmVoxFile,'rb');
    matchSegmVoxSize = fread(fileID,3,'single');
    matchSegmOrigin = fread(fileID,3,'single');
    segm2model = reshape(fread(fileID,16,'single'),4,4)';
    matchSegmVox = fread(fileID,'single');
    fclose(fileID);
    matchSegmVox = reshape(matchSegmVox,matchSegmVoxSize');

    % Generate keypoints
    numKeypoints = 1000;
    numKeypoints = min(floor(size(find(matchSegmVox > 0.8),1)/50)*50,numKeypoints);
    randSurfInd = randsample(find(matchSegmVox > 0.8),numKeypoints);
    [segmKeypointsGridX,segmKeypointsGridY,segmKeypointsGridZ] = ind2sub(size(matchSegmVox),randSurfInd);
    segmKeypointsGrid = [segmKeypointsGridX,segmKeypointsGridY,segmKeypointsGridZ];
    segmKeypointsCam = (segmKeypointsGrid-1).*0.005+repmat(matchSegmOrigin',size(segmKeypointsGrid,1),1);
    
    % Save keypoints
    fileID = fopen(fullfile(scenarioPath,'segmentation.keypoints.bin'),'wb');
    fwrite(fileID,numKeypoints,'single');
    segmKeypointsCamFlipped = segmKeypointsCam';
    fwrite(fileID,segmKeypointsCamFlipped(:),'single');
    fclose(fileID);
end

% Compute keypoints for object models
objectFiles = dir(fullfile(dataPath,'objects/*.ply'));
for objectIdx = 1:length(objectFiles)
    objectName = objectFiles(objectIdx).name(1:(end-4));
    fprintf('%s\n',objectName);

    % Load object model point cloud and TDF voxel grid
    matchModelVoxFile = fullfile(dataPath,'objects',sprintf('%s.TDF.bin',objectName));
    modelPointCloud = pcread(fullfile(dataPath,'objects',sprintf('%s.ply',objectName)));
    fileID = fopen(matchModelVoxFile,'rb');
    matchModelVoxSize = fread(fileID,3,'single');
    matchModelOrigin = fread(fileID,3,'single');
    matchModelVox = fread(fileID,'single');
    fclose(fileID);
    matchModelVox = reshape(matchModelVox,matchModelVoxSize');
    
    % Geenrate keypoints
    numKeypoints = 1000;
    numKeypoints = min(floor(size(find(matchModelVox > 0.8),1)/50)*50,numKeypoints);
    randSurfInd = randsample(find(matchModelVox > 0.8),numKeypoints);
    [modelKeypointsGridX,modelKeypointsGridY,modelKeypointsGridZ] = ind2sub(size(matchModelVox),randSurfInd);
    modelKeypointsGrid = [modelKeypointsGridX,modelKeypointsGridY,modelKeypointsGridZ];
    modelKeypointsCam = (modelKeypointsGrid-1).*0.005+repmat(matchModelOrigin',size(modelKeypointsGrid,1),1);
    
    % Save keypoints
    fileID = fopen(fullfile(dataPath,'objects',sprintf('%s.keypoints.bin',objectName)),'wb');
    fwrite(fileID,numKeypoints,'single');
    modelKeypointsCamFlipped = modelKeypointsCam';
    fwrite(fileID,modelKeypointsCamFlipped(:),'single');
    fclose(fileID);
end

