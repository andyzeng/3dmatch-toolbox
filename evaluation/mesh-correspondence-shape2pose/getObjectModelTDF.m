% Compute TDF voxel grid volumes of object models
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
dataPath = '../../data/shape2pose/bikes';

objectFiles = dir(fullfile(dataPath,'*.raw'));
for objectIdx = 1:length(objectFiles)
    objectPrefix = objectFiles(objectIdx).name(1:(end-4));
    fprintf('%s\n',objectPrefix);
    objectSizeFile = fullfile(dataPath,sprintf('%s.size',objectPrefix));
    objectRawFile = fullfile(dataPath,sprintf('%s.raw',objectPrefix));
    
    % Load voxel grid dimensions of distance transform from GAPS
    fileID = fopen(objectSizeFile,'r');
    objectVoxelGridSize = fscanf(fileID,'Float32(%d,%d,%d,%d)\n');
    fclose(fileID);
    
    % Load distance transform voxel grid from GAPS
    fid = fopen(objectRawFile,'rb');
    objectRawData = fread(fid,'single');
    fclose(fid);
    objectVoxelGrid = reshape(objectRawData,objectVoxelGridSize(1),objectVoxelGridSize(2),objectVoxelGridSize(3));

    % Truncate voxel grid values
    truncationSize = 5; % in voxels
    objectVoxelGrid(find(abs(objectVoxelGrid) > truncationSize)) = truncationSize;
    objectVoxelGrid = objectVoxelGrid./truncationSize;
    
    % Flip voxel grid values
    objectVoxelGridTDF = 1 - objectVoxelGrid;

    % Find all surface keypoints
    [gridX,gridY,gridZ] = ndgrid(1:objectVoxelGridSize(1),1:objectVoxelGridSize(2),1:objectVoxelGridSize(3));
    surfaceIdx = find(abs(objectVoxelGridTDF) > 0.8);
    keypoints = [gridX(surfaceIdx),gridY(surfaceIdx),gridZ(surfaceIdx)]-1;
    
    % Trailing keypoints
    keypoints = [keypoints;repmat(keypoints(end,:),50-mod(size(keypoints,1),50),1)];
   
    keypointsFile = fullfile(dataPath,sprintf('%s.keypoints.bin',objectPrefix));
    TDFVoxelGridFile = fullfile(dataPath,sprintf('%s.TDF.bin',objectPrefix));
    
    % Save surface keypoints to binary file
    fid = fopen(keypointsFile,'wb');
    fwrite(fid,size(keypoints,1),'single');
    flippedKeypoints = keypoints';
    fwrite(fid,flippedKeypoints(:),'single');
    fclose(fid);

    % Save TDF voxel grid volume to TDF file
    fid = fopen(TDFVoxelGridFile,'wb');
    fwrite(fid,size(objectVoxelGridTDF),'single');
    fwrite(fid,objectVoxelGridTDF(:),'single');
    fclose(fid);
end
