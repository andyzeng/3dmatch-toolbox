% Compute voxel grid volume of accurate TDF values for object models
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
objModelsPath = '../../data/apc/objects'; % Path to object model point clouds
voxelSize = 0.005; % in meters
voxelMargin = 5; % in voxels

objModelDir = dir(fullfile(objModelsPath,'*.ply'));
for objModelIdx = 1:length(objModelDir)
    objModelName = objModelDir(objModelIdx).name(1:(end-4));
    fprintf('%s\n',objModelName);
    
    % Load object model point cloud
    objModelPointCloud = pcread(fullfile(objModelsPath,sprintf('%s.ply',objModelName)));
    objModelPointCloud = pcdownsample(objModelPointCloud,'gridAverage',voxelSize/2);
    objModelPointCloud = pcdenoise(objModelPointCloud);
    
    % Compute object model TDF voxel grid
    objModelPts = objModelPointCloud.Location';
    objModelRangeX = [objModelPointCloud.XLimits(1)-voxelSize*20, objModelPointCloud.XLimits(2)+voxelSize*20];
    objModelRangeY = [objModelPointCloud.YLimits(1)-voxelSize*20, objModelPointCloud.YLimits(2)+voxelSize*20];
    objModelRangeZ = [objModelPointCloud.ZLimits(1)-voxelSize*20, objModelPointCloud.ZLimits(2)+voxelSize*20];
    objModelTDF = pointCloud2TDF(objModelPts,objModelRangeX,objModelRangeY,objModelRangeZ,voxelSize,voxelMargin);
    
%     apcVisualizeTDF(objModelITUDF,'model');
%     [objModelKeyptsGrid,objModelKeyptsFeat] = getKeyptFeat(objModelITUDF);
%     objModelKeyptsCamX = (objModelRangeX(1)-tudfVoxelSize/2)+tudfVoxelSize*objModelKeyptsGrid(:,1);
%     objModelKeyptsCamY = (objModelRangeY(1)-tudfVoxelSize/2)+tudfVoxelSize*objModelKeyptsGrid(:,2);
%     objModelKeyptsCamZ = (objModelRangeZ(1)-tudfVoxelSize/2)+tudfVoxelSize*objModelKeyptsGrid(:,3);
%     objModelKeyptsCam = [objModelKeyptsCamX,objModelKeyptsCamY,objModelKeyptsCamZ];
%     objectVox.voxelSize = tudfVoxelSize;
%     objectVox.voxelMargin = tudfVoxelMargin;
%     objectVox.ITUDF = objModelITUDF;
%     objectVox.surfacePointsGridCoord = objModelKeyptsGrid;
%     objectVox.surfacePointsObjCoord = objModelKeyptsCam;
%     objectVox.SurfacePointsFeat = objModelKeyptsFeat;
%     save(fullfile(objModelsPath,sprintf('%s.vox.mat',objModelName)),'objectVox');
    
    % Save TDF voxel grid
    objModelOrigin = [objModelRangeX(1)+voxelSize/2,objModelRangeY(1)+voxelSize/2,objModelRangeZ(1)+voxelSize/2];
    fp = fopen(fullfile(objModelsPath,sprintf('%s.TDF.bin',objModelName)),'wb');
    fwrite(fp,size(objModelTDF),'single');
    fwrite(fp,objModelOrigin,'single');
    fwrite(fp,objModelTDF,'single');
    fclose(fp);
end