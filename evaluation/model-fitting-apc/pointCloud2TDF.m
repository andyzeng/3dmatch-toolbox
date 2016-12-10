function voxelGridTDF = pointCloud2TDF(points,xRange,yRange,zRange,voxelSize,voxelMargin)
% Given a point cloud, compute a voxel grid of TDF values (warning: slow!)
% Works fine for small point clouds - uses in-house Matlab functions
%
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Compute voxel grid
[gridX,gridY,gridZ] = ndgrid((xRange(1)+voxelSize/2):voxelSize:(xRange(2)-voxelSize/2), ...
                             (yRange(1)+voxelSize/2):voxelSize:(yRange(2)-voxelSize/2), ...
                             (zRange(1)+voxelSize/2):voxelSize:(zRange(2)-voxelSize/2));

% Build KD-tree and do 1-NN search                         
modelKDT = KDTreeSearcher(points');
[nnIdx,nnDist] = knnsearch(modelKDT,[gridX(:),gridY(:),gridZ(:)]);

% Reshape values into voxel grid
voxelGridTDF = reshape(nnDist,size(gridX));

% Apply truncation
voxelGridTDF = 1.0 - min(voxelGridTDF./(voxelSize*voxelMargin),1.0);

end

