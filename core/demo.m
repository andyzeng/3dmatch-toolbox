clear all;

% Add dependencies (for RANSAC-based rigid transform estimation)
addpath(genpath('external'));

fragment1PointCloudFile = '../data/sample/3dmatch-demo/single-depth-1.ply';
fragment1KeypointsFile = 'fragment-1.keypts.bin';
fragment1DescriptorsFile = 'fragment-1.desc.3dmatch.bin';

fragment2PointCloudFile = '../data/sample/3dmatch-demo/single-depth-2.ply';
fragment2KeypointsFile = 'fragment-2.keypts.bin';
fragment2DescriptorsFile = 'fragment-2.desc.3dmatch.bin';

% Load fragment point clouds
fragment1PointCloud = pcread(fragment1PointCloudFile);
fragment2PointCloud = pcread(fragment2PointCloudFile);

% Load keypoints of fragment 1
fid = fopen(fragment1KeypointsFile,'rb');
numFragment1Keypoints = fread(fid,1,'single');
fragment1Keypoints = fread(fid,'single');
fragment1Keypoints = reshape(fragment1Keypoints,3,numFragment1Keypoints)';
fclose(fid);

% Load 3DMatch feature descriptors for keypoints of fragment 1
fid = fopen(fragment1DescriptorsFile,'rb');
fragment1DescriptorData = fread(fid,'single');
fragment1NumDescriptors = fragment1DescriptorData(1);
fragment1DescriptorSize = fragment1DescriptorData(2);
fragment1Descriptors = reshape(fragment1DescriptorData(3:end),fragment1DescriptorSize,fragment1NumDescriptors)';
fclose(fid);

% Load keypoints of fragment 2
fid = fopen(fragment2KeypointsFile,'rb');
numFragment2Keypoints = fread(fid,1,'single');
fragment2Keypoints = fread(fid,'single');
fragment2Keypoints = reshape(fragment2Keypoints,3,numFragment2Keypoints)';
fclose(fid);

% Load 3DMatch feature descriptors for keypoints of fragment 2
fid = fopen(fragment2DescriptorsFile,'rb');
fragment2DescriptorData = fread(fid,'single');
fragment2NumDescriptors = fragment2DescriptorData(1);
fragment2DescriptorSize = fragment2DescriptorData(2);
fragment2Descriptors = reshape(fragment2DescriptorData(3:end),fragment2DescriptorSize,fragment2NumDescriptors)';
fclose(fid);

% Find mutually closest keypoints in 3DMatch descriptor space
fprintf('Finding mutually closest points in 3DMatch descriptor space...\n');
fragment2KDT = KDTreeSearcher(fragment2Descriptors);
fragment1KDT = KDTreeSearcher(fragment1Descriptors);
fragment1NNIdx = knnsearch(fragment2KDT,fragment1Descriptors);
fragment2NNIdx = knnsearch(fragment1KDT,fragment2Descriptors);
fragment2MatchIdx = find((1:size(fragment2NNIdx,1))' == fragment1NNIdx(fragment2NNIdx));
fragment2MatchKeypoints = fragment2Keypoints(fragment2MatchIdx,:);
fragment1MatchKeypoints = fragment1Keypoints(fragment2NNIdx(fragment2MatchIdx),:);

% Estimate rigid transformation with RANSAC to align fragment 2 to fragment 1
fprintf('Running RANSAC to estimate rigid transformation...');
[estimateRt,inlierIdx] = ransacfitRt([fragment1MatchKeypoints';fragment2MatchKeypoints'], 0.05, 0);
estimateRt = [estimateRt;[0,0,0,1]];
fprintf('Estimated rigid transformation to align fragment 2 to fragment 1:\n');
fprintf('\t %15.8e\t %15.8e\t %15.8e\t %15.8e\t\n',estimateRt');

fragment2Points = estimateRt(1:3,1:3) * fragment2PointCloud.Location' + repmat(estimateRt(1:3,4),1,size(fragment2PointCloud.Location',2));
fragment1Points = fragment1PointCloud.Location';

% Compute alignment percentage (for loop closure detection)
fprintf('Computing surface alignment overlap...\n');
[nnIdx,sqrDists] = multiQueryKNNSearchImpl(pointCloud(fragment2Points'),fragment1Points',1);
dists = sqrt(sqrDists);
ratioAligned = sum(dists < 0.05)/size(fragment1Points,2);
fprintf('Estimated surface overlap: %.1f%%\n',ratioAligned*100);

% Visualize alignment results
pcwrite(pointCloud([fragment2Points';fragment1Points'],'Color',[repmat(uint8([0,0,255]),size(fragment2Points,2),1);repmat(uint8([255,0,0]),size(fragment1Points,2),1)]),'result','PLYformat','binary');
fprintf('Generated visualization of alignment! See results in result.ply\n');

%% Code snippet to generate point cloud from single depth map
% depth = imread('/home/andyz/3dmatch/legacy/data/test/7-scenes-redkitchen/seq-02/frame-000350.depth.png');
% depth = double(depth)./1000;
% K = [585,0,320;0,585,240;0,0,1];
% [pixX,pixY] = meshgrid(1:640,1:480);
% camX = (pixX-K(1,3)).*depth/K(1,1);
% camY = (pixY-K(2,3)).*depth/K(2,2);
% camZ = depth;
% validDepth = find((camZ ~= 0) & (camZ < 6));
% camPts = [camX(validDepth),camY(validDepth),camZ(validDepth)]';
% pcwrite(pointCloud(single(camPts')),'single-depth-1','PLYFormat','binary');
% 
% depth = imread('/home/andyz/3dmatch/legacy/data/test/7-scenes-redkitchen/seq-06/frame-000270.depth.png');
% depth = double(depth)./1000;
% K = [585,0,320;0,585,240;0,0,1];
% [pixX,pixY] = meshgrid(1:640,1:480);
% camX = (pixX-K(1,3)).*depth/K(1,1);
% camY = (pixY-K(2,3)).*depth/K(2,2);
% camZ = depth;
% validDepth = find((camZ ~= 0) & (camZ < 6));
% camPts = [camX(validDepth),camY(validDepth),camZ(validDepth)]';
% pcwrite(pointCloud(single(camPts')),'single-depth-2','PLYFormat','binary');

%% Code snippet for visualizing TDF voxel grid (reads tdf.bin generated from demo.cu)
% fid = fopen('tdf.bin','rb');
% voxelGridSize = fread(fid,3,'single');
% voxelGridOrigin = fread(fid,3,'single');
% voxelSize = fread(fid,1,'single');
% truncMargin = fread(fid,1,'single');
% voxelGridTDFData = fread(fid,'single');
% fclose(fid);
% voxelGridTDF = reshape(voxelGridTDFData,voxelGridSize(1),voxelGridSize(2),voxelGridSize(3));
% 
% [gridX,gridY,gridZ] = ndgrid(1:size(voxelGridTDF,1),1:size(voxelGridTDF,2),1:size(voxelGridTDF,3));
% voxelGridPts = [gridX(:),gridY(:),gridZ(:)];
% % voxelGridPts = ((voxelGridPts-1).*voxelSize) + repmat(voxelGridOrigin'+[0.005,0.005,0.005],size(voxelGridPts,1),1);
% 
% % Visualize
% surfaceIdx = find(voxelGridTDF > 0);
% surfacePts = voxelGridPts(surfaceIdx,:);
% surfaceDists = voxelGridTDF(surfaceIdx);
% cmap = jet;
% colorPts = cmap(round(surfaceDists*63)+1,:);
% pcwrite(pointCloud(surfacePts,'Color',colorPts),'tdf','PLYFormat','binary');
