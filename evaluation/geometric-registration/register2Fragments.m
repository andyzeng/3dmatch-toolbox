function [estimateRt,numInliers,inlierRatio,ratioAligned] = register2Fragments(scenePath,intermPath,fragment1Name,fragment2Name,descriptorName)
% Run RANSAC-based pose estimation between two sets of 3D keypoints and
% their 3DMatch descriptors (of scene fragments)
%
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Load fragment point clouds
fragment1PointCloud = pcread(fullfile(scenePath,sprintf('%s.ply',fragment1Name)));
fragment2PointCloud = pcread(fullfile(scenePath,sprintf('%s.ply',fragment2Name)));

% Load keypoints of fragment 1
fid = fopen(fullfile(intermPath,sprintf('%s.keypts.bin',fragment1Name)),'rb');
numFragment1Keypoints = fread(fid,1,'single');
fragment1Keypoints = fread(fid,'single');
fragment1Keypoints = reshape(fragment1Keypoints,3,numFragment1Keypoints)';
fclose(fid);

% Load 3DMatch feature descriptors for keypoints of fragment 1
fid = fopen(fullfile(intermPath,sprintf('%s.desc.%s.bin',fragment1Name,descriptorName)),'rb');
fragment1DescriptorData = fread(fid,'single');
fragment1NumDescriptors = fragment1DescriptorData(1);
fragment1DescriptorSize = fragment1DescriptorData(2);
fragment1Descriptors = reshape(fragment1DescriptorData(3:end),fragment1DescriptorSize,fragment1NumDescriptors)';
fclose(fid);

% Load keypoints of fragment 2
fid = fopen(fullfile(intermPath,sprintf('%s.keypts.bin',fragment2Name)),'rb');
numFragment2Keypoints = fread(fid,1,'single');
fragment2Keypoints = fread(fid,'single');
fragment2Keypoints = reshape(fragment2Keypoints,3,numFragment2Keypoints)';
fclose(fid);

% Load 3DMatch feature descriptors for keypoints of fragment 2
fid = fopen(fullfile(intermPath,sprintf('%s.desc.%s.bin',fragment2Name,descriptorName)),'rb');
fragment2DescriptorData = fread(fid,'single');
fragment2NumDescriptors = fragment2DescriptorData(1);
fragment2DescriptorSize = fragment2DescriptorData(2);
fragment2Descriptors = reshape(fragment2DescriptorData(3:end),fragment2DescriptorSize,fragment2NumDescriptors)';
fclose(fid);

% Find mutually closest keypoints in 3DMatch feature descriptor space
fragment2KDT = KDTreeSearcher(fragment2Descriptors);
fragment1KDT = KDTreeSearcher(fragment1Descriptors);
fragment1NNIdx = knnsearch(fragment2KDT,fragment1Descriptors);
fragment2NNIdx = knnsearch(fragment1KDT,fragment2Descriptors);
fragment2MatchIdx = find((1:size(fragment2NNIdx,1))' == fragment1NNIdx(fragment2NNIdx));
fragment2MatchKeypoints = fragment2Keypoints(fragment2MatchIdx,:);
fragment1MatchKeypoints = fragment1Keypoints(fragment2NNIdx(fragment2MatchIdx),:);

% Estimate initial transformation with RANSAC to align fragment 2 keypoints to fragment 1 keypoints
try
    [estimateRt,inlierIdx] = ransacfitRt([fragment1MatchKeypoints';fragment2MatchKeypoints'], 0.05, 0);
    estimateRt = [estimateRt;[0,0,0,1]];
catch
    fprintf('Error: not enough mutually matching keypoints!\n');
    estimateRt = eye(4);
    inlierIdx = [];
end

fragment2Points = estimateRt(1:3,1:3) * fragment2PointCloud.Location' + repmat(estimateRt(1:3,4),1,size(fragment2PointCloud.Location',2));
fragment1Points = fragment1PointCloud.Location';

% % Refine rigid transformation with ICP
% if useGPU
%     [tform,movingReg,icpRmse] = pcregrigidGPU(pcdownsample(pointCloud(fragment1Points'),'gridAverage',0.01),pcdownsample(pointCloud(fragment2Points'),'gridAverage',0.01),'InlierRatio',0.3,'Verbose',true,'Tolerance',[0.0001,0.00009],'Extrapolate',true,'MaxIterations',50);
% else
%     [tform,movingReg,icpRmse] = pcregrigid(pcdownsample(pointCloud(fragment1Points'),'gridAverage',0.01),pcdownsample(pointCloud(fragment2Points'),'gridAverage',0.01),'InlierRatio',0.3,'Verbose',false,'Tolerance',[0.0001,0.00009],'Extrapolate',true,'MaxIterations',50); %,
% end
% icpRt = inv(tform.T');
% fragment2Points = icpRt(1:3,1:3) * fragment2Points + repmat(icpRt(1:3,4),1,size(fragment2Points,2));
% estimateRt = icpRt * estimateRt;

resultsPath = fullfile(intermPath,sprintf('%s-registration-results',descriptorName));

if ~exist(resultsPath)
    mkdir(resultsPath);
end

% % Visualize alignment result
% visualizationPath = fullfile(resultsPath,sprintf('%s-%s',fragment1Name,fragment2Name));
% pcwrite(pointCloud([fragment2Points';fragment1Points'],'Color',[repmat(uint8([0,0,255]),size(fragment2Points,2),1);repmat(uint8([255,0,0]),size(fragment1Points,2),1)]),visualizationPath,'PLYformat','binary');

% Compute alignment percentage
ratioAligned = zeros(1,2);
[nnIdx,sqrDists] = multiQueryKNNSearchImpl(pointCloud(fragment2Points'),fragment1Points',1);
dists = sqrt(sqrDists);
ratioAligned(1) = sum(dists < 0.05)/size(fragment1Points,2); % relative overlap on first fragment
[nnIdx,sqrDists] = multiQueryKNNSearchImpl(pointCloud(fragment1Points'),fragment2Points',1);
dists = sqrt(sqrDists);
ratioAligned(2) = sum(dists < 0.05)/size(fragment2Points,2); % relative overlap on second fragment

% Compute several additional heuristics for loop closure detection
numInliers = length(inlierIdx);
inlierRatio = numInliers/length(fragment2MatchIdx);

end











