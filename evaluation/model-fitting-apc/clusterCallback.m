function clusterCallback(jobID)
% Predict object pose between object model and segmentation point cloud by
% running RANSAC-based geometric registration between two sets of 3D
% keypoints and their descriptors
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
descriptorName = '3dmatch';
dataPath = '../../data/apc';
addpath(genpath('../../core/external'));

% List scenarios
scenarioIdx = jobID - 1;
scenarioList = dir(fullfile(dataPath,'scenarios/00*'));
if scenarioIdx >= length(scenarioList)
    return;
end
scenarioPath = fullfile(dataPath,'scenarios',sprintf('%06d',scenarioIdx));
fprintf('%s\n',scenarioPath);

% Check if pose prediction file already exists
predictedRtFile = fullfile(scenarioPath,sprintf('ransac.%s.model2segm.txt',descriptorName));
if exist(predictedRtFile,'file')
    return;
end

fileID = fopen(fullfile(scenarioPath,'info.txt'),'r');
sceneName = fscanf(fileID,'scene: %s\n');
objectName = fscanf(fileID,'object: %s\n');
fclose(fileID);

% Load segmentation from scene point cloud and its TDF voxel volume
segmPointCloud = pcread(fullfile(scenarioPath,'segmentation.ply'));
matchSegmVoxFile = fullfile(scenarioPath,'segmentation.TDF.bin');
fileID = fopen(matchSegmVoxFile,'rb');
matchSegmVoxSize = fread(fileID,3,'single');
matchSegmOrigin = fread(fileID,3,'single');
segm2model = reshape(fread(fileID,16,'single'),4,4)';
matchSegmVox = fread(fileID,'single');
fclose(fileID);
matchSegmVox = reshape(matchSegmVox,matchSegmVoxSize');

% Load object model point cloud and its TDF voxel volume
matchModelVoxFile = fullfile(dataPath,'objects',sprintf('%s.TDF.bin',objectName));
modelPointCloud = pcread(fullfile(dataPath,'objects',sprintf('%s.ply',objectName)));
fileID = fopen(matchModelVoxFile,'rb');
matchModelVoxSize = fread(fileID,3,'single');
matchModelOrigin = fread(fileID,3,'single');
matchModelVox = fread(fileID,'single');
fclose(fileID);
matchModelVox = reshape(matchModelVox,matchModelVoxSize');

% Load segmentation keypoints
segmKeypointsFile = fullfile(scenarioPath,'segmentation.keypoints.bin');
fileID = fopen(segmKeypointsFile,'rb');
segmNumKeypoints = fread(fileID,1,'single');
segmKeypointsCamData = fread(fileID,'single');
fclose(fileID);
segmKeypointsCam = reshape(segmKeypointsCamData,3,segmNumKeypoints)';

% Load model keypoints
modelKeypointsFile = fullfile(dataPath,'objects',sprintf('%s.keypoints.bin',objectName));
fileID = fopen(modelKeypointsFile,'rb');
modelNumKeypoints = fread(fileID,1,'single');
modelKeypointsCamData = fread(fileID,'single');
fclose(fileID);
modelKeypointsCam = reshape(modelKeypointsCamData,3,modelNumKeypoints)';

% Load segmentation keypoint descriptors
fileID = fopen(fullfile(scenarioPath,sprintf('segmentation.keypoints.%s.descriptors.bin',descriptorName)),'rb');
segmKeypointDescriptorsData = fread(fileID,'single');
segmNumKeypoints = segmKeypointDescriptorsData(1);
segmDescriptorSize = segmKeypointDescriptorsData(2);
fclose(fileID);
segmKeypointDescriptors = reshape(segmKeypointDescriptorsData(3:end),segmDescriptorSize,segmNumKeypoints)';

% Load model keypoint descriptors
fileID = fopen(fullfile(dataPath,'objects',sprintf('%s.keypoints.%s.descriptors.bin',objectName,descriptorName)),'rb');
modelKeypointDescriptorsData = fread(fileID,'single');
modelNumKeypoints = modelKeypointDescriptorsData(1);
modelDescriptorSize = modelKeypointDescriptorsData(2);
fclose(fileID);
modelKeypointDescriptors = reshape(modelKeypointDescriptorsData(3:end),modelDescriptorSize,modelNumKeypoints)';

% Find mutually closest keypoints in descriptor space
modelKDT = KDTreeSearcher(modelKeypointDescriptors);
segmKDT = KDTreeSearcher(segmKeypointDescriptors);
nnSegmIdx = knnsearch(modelKDT,segmKeypointDescriptors);
nnModelIdx = knnsearch(segmKDT,modelKeypointDescriptors);
matchModelIdx = find((1:size(nnModelIdx,1))' == nnSegmIdx(nnModelIdx));
modelKeypointsCam = modelKeypointsCam(matchModelIdx,:);
segmKeypointsCam = segmKeypointsCam(nnModelIdx(matchModelIdx),:);

% Use RANSAC to estimate rigid transformation from model to segmentation
try
    [estModel2SegmRt, inliers] = ransacfitRt([segmKeypointsCam';modelKeypointsCam'], 0.05, 0);
catch
    estModel2SegmRt = [1,0,0,0;0,1,0,0;0,0,1,0];
    fprintf('   Broken\n');
end
estModel2SegmRt = [estModel2SegmRt;[0,0,0,1]];

% % Visualize results
% checkObjModelPts = estModel2SegmRt(1:3,1:3) * modelPointCloud.Location' + repmat(estModel2SegmRt(1:3,4),1,size(modelPointCloud.Location',2));
% checkObjSegmPts = segmPointCloud.Location';
% pcwrite(pointCloud([checkObjModelPts';checkObjSegmPts'],'Color',[repmat(uint8([0,0,255]),size(checkObjModelPts,2),1);repmat(uint8([255,0,0]),size(checkObjSegmPts,2),1)]),'check','PLYformat','binary');

% Save results to pose prediction file
fileID = fopen(predictedRtFile,'w');
for i = 1:4
    fprintf(fileID,'%15.8e\t',estModel2SegmRt(i,:));
    fprintf(fileID,'\n');
end
fclose(fileID);

end













