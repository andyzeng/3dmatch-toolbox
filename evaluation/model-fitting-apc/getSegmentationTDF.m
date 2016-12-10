% Compute voxel grid volume of accurate TDF values for segmentation point
% cloud
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
benchmarkPath = '/home/andyz/apc/toolbox/data/benchmark'; % Path to Shelf & Tote benchmark files
sceneDataPath = '../../data/apc/scenes'; % Path to scene data
voxelSize = 0.005; % in meters
voxelMargin = 5; % in voxels

load(fullfile(benchmarkPath,'labels.mat'));

% trainSegmPathList = {};
% fid = fopen('sceneList.txt','w');

for labelIdx = 1:length(labels)
    fprintf('%d/%d\n',labelIdx,length(labels));
    tmpLabel = labels{labelIdx};
    
    % Get ground truth 6D pose label
    extWorld2Bin = inv(dlmread(fullfile(benchmarkPath,tmpLabel.sceneName,'cam.info.txt'),'\t',[21,0,24,3]));
    extModel2Segm = extWorld2Bin*tmpLabel.objectPose;
    extSegm2Model = inv(extModel2Segm);
    
    % If there are multiple instances, choose the corresponding instance
    segmDir = dir(fullfile(sceneDataPath,tmpLabel.sceneName,sprintf('%s.*.segm.ply',tmpLabel.objectName)));
    if isempty(segmDir)
        continue;
    elseif length(segmDir) == 1
        segmPath = fullfile(sceneDataPath,tmpLabel.sceneName,segmDir.name(1:(end-9)));
    else
        gtTrans = extModel2Segm(1:3,4);
        closestInstanceIdx = 0;
        closestInstanceDist = Inf;
        for instanceIdx = 1:length(segmDir)
            instancePointCloud = pcread(fullfile(sceneDataPath,tmpLabel.sceneName,segmDir(instanceIdx).name));
            instanceDist = sqrt(sum((gtTrans-mean(instancePointCloud.Location)').^2));
            if instanceDist < closestInstanceDist
                closestInstanceIdx = instanceIdx;
                closestInstanceDist = instanceDist;
            end
        end
        segmPath = fullfile(sceneDataPath,tmpLabel.sceneName,segmDir(closestInstanceIdx).name(1:(end-9)));
    end
    
    % Load segmentation of scene point cloud
    segmPointCloud = pcread(sprintf('%s.segm.ply',segmPath));
    segmPointCloud = pcdownsample(segmPointCloud,'gridAverage',voxelSize/2);
    segmPointCloud = pcdenoise(segmPointCloud);

    % Generate TDF voxel grid of segmentation
    objSegmPts = segmPointCloud.Location';
    objSegmRangeX = [segmPointCloud.XLimits(1)-voxelSize*20, segmPointCloud.XLimits(2)+voxelSize*20];
    objSegmRangeY = [segmPointCloud.YLimits(1)-voxelSize*20, segmPointCloud.YLimits(2)+voxelSize*20];
    objSegmRangeZ = [segmPointCloud.ZLimits(1)-voxelSize*20, segmPointCloud.ZLimits(2)+voxelSize*20];
    objSegmTDF = pointCloud2TDF(objSegmPts,objSegmRangeX,objSegmRangeY,objSegmRangeZ,voxelSize,voxelMargin);
%     showITUDF(objSegmITUDF,'segm');
    objSegmOrigin = [objSegmRangeX(1)+voxelSize/2,objSegmRangeY(1)+voxelSize/2,objSegmRangeZ(1)+voxelSize/2];

%     if exist(sprintf('%s.segm.vox.bin',segmPath),'file')
%         delete(sprintf('%s.segm.vox.bin',segmPath));
%     end
    
    % Write TDF to disk
    fp = fopen(sprintf('%s.segmentation.TDF.bin',segmPath),'wb');
    fwrite(fp,size(objSegmTDF),'single');
    fwrite(fp,objSegmOrigin,'single');
    tmpRt = extSegm2Model';
    fwrite(fp,tmpRt,'single');
    fwrite(fp,objSegmTDF,'single');
    fclose(fp);
    
%     fprintf(fid,'1 %s.segmentation.TDF.bin 1\n',segmPath);
end

% fclose(fid);

% trainSegmPathList = trainSegmPathList';




% objModelsPath = './apc/objects';
% objModelDir = dir(fullfile(objModelsPath,'*.ply'));
% for objModelIdx = 1:length(objModelDir)
%     objModelName = objModelDir(objModelIdx).name(1:(end-4));
%     segmPointCloud = pcread(fullfile(objModelsPath,sprintf('%s.ply',objModelName)));
%     segmPointCloud = pcdownsample(segmPointCloud,'gridAverage',tudfVoxelSize/2);
%     segmPointCloud = pcdenoise(segmPointCloud);
%     
%     objSegmPts = segmPointCloud.Location';
%     objSegmRangeX = [segmPointCloud.XLimits(1)-tudfVoxelSize*20, segmPointCloud.XLimits(2)+tudfVoxelSize*20];
%     objSegmRangeY = [segmPointCloud.YLimits(1)-tudfVoxelSize*20, segmPointCloud.YLimits(2)+tudfVoxelSize*20];
%     objSegmRangeZ = [segmPointCloud.ZLimits(1)-tudfVoxelSize*20, segmPointCloud.ZLimits(2)+tudfVoxelSize*20];
%     objSegmITUDF = getITUDFVox(objSegmPts,objSegmRangeX,objSegmRangeY,objSegmRangeZ,tudfVoxelSize,tudfVoxelMargin);
%     showITUDF(objSegmITUDF,'model');
% 
%     [objModelKeyptsGrid,objModelKeyptsFeat] = getKeyptFeat(objSegmITUDF);
%     objModelKeyptsCamX = (objSegmRangeX(1)-tudfVoxelSize/2)+tudfVoxelSize*objModelKeyptsGrid(:,1);
%     objModelKeyptsCamY = (objSegmRangeY(1)-tudfVoxelSize/2)+tudfVoxelSize*objModelKeyptsGrid(:,2);
%     objModelKeyptsCamZ = (objSegmRangeZ(1)-tudfVoxelSize/2)+tudfVoxelSize*objModelKeyptsGrid(:,3);
%     objModelKeyptsCam = [objModelKeyptsCamX,objModelKeyptsCamY,objModelKeyptsCamZ];
% 
%     objectVox.voxelSize = tudfVoxelSize;
%     objectVox.voxelMargin = tudfVoxelMargin;
%     objectVox.ITUDF = objSegmITUDF;
%     objectVox.surfacePointsGridCoord = objModelKeyptsGrid;
%     objectVox.surfacePointsObjCoord = objModelKeyptsCam;
%     objectVox.SurfacePointsFeat = objModelKeyptsFeat;
%     save(fullfile(objModelsPath,sprintf('%s.vox.mat',objModelName)),'objectVox');
%     
%     fp = fopen(fullfile(objModelsPath,sprintf('%s.vox.bin',objModelName)),'wb');
%     fwrite(fp,size(objSegmITUDF),'single');
%     fwrite(fp,objSegmITUDF,'single');
%     fclose(fp);
% 
% end