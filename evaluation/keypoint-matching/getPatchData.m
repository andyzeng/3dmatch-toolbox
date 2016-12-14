function [colorPatch,depthPatch,voxelGridTDF] = getPatchData(pointData,voxelGridPatchRadius,voxelSize,voxelMargin)
%GETPATCHDATA Summary of this function goes here
%   Detailed explanation goes here

    % Get color/depth image patches
    colorIm = imread(sprintf('%s.color.png',pointData.framePath));
    depthIm = double(imread(sprintf('%s.depth.png',pointData.framePath)))./1000;
    depthIm(find(depthIm > 6)) = 0;
    colorPatch = colorIm(min(pointData.bboxRangePixels(2,:)):max(pointData.bboxRangePixels(2,:)),min(pointData.bboxRangePixels(1,:)):max(pointData.bboxRangePixels(1,:)),:);
    depthPatch = depthIm(min(pointData.bboxRangePixels(2,:)):max(pointData.bboxRangePixels(2,:)),min(pointData.bboxRangePixels(1,:)):max(pointData.bboxRangePixels(1,:)));
    
     % Get TDF voxel grid local patches
    [pixX,pixY] = meshgrid(min(pointData.bboxRangePixels(1,:)):max(pointData.bboxRangePixels(1,:)),min(pointData.bboxRangePixels(2,:)):max(pointData.bboxRangePixels(2,:)));
    camX = (pixX-pointData.camK(1,3)).*depthPatch./pointData.camK(1,1);
    camY = (pixY-pointData.camK(2,3)).*depthPatch./pointData.camK(2,2);
    camZ = depthPatch;
    validDepth = find(depthPatch > 0);
    camPts = [camX(:),camY(:),camZ(:)];
    camPts = camPts(validDepth,:);
%     pcwrite(pointCloud(camPts),'test','PLYFormat','binary');
    [gridPtsCamX,gridPtsCamY,gridPtsCamZ] = ndgrid((pointData.camCoords(1)-voxelGridPatchRadius*voxelSize+voxelSize/2):voxelSize:(pointData.camCoords(1)+voxelGridPatchRadius*voxelSize-voxelSize/2), ...
                                                   (pointData.camCoords(2)-voxelGridPatchRadius*voxelSize+voxelSize/2):voxelSize:(pointData.camCoords(2)+voxelGridPatchRadius*voxelSize-voxelSize/2), ...
                                                   (pointData.camCoords(3)-voxelGridPatchRadius*voxelSize+voxelSize/2):voxelSize:(pointData.camCoords(3)+voxelGridPatchRadius*voxelSize-voxelSize/2));
    gridPtsCam = [gridPtsCamX(:),gridPtsCamY(:),gridPtsCamZ(:)];
    
    % Use 1-NN search to get TDF values
    [knnIdx,knnDist] = knnsearch(camPts,gridPtsCam);
    TDFValues = knnDist./voxelMargin; % truncate
    TDFValues(find(TDFValues > 1)) = 1;
    TDFValues = 1-TDFValues; % flip
    voxelGridTDF = reshape(TDFValues,size(gridPtsCamX));
    
    % Convert to floats to save space
    voxelGridTDF = single(voxelGridTDF);
    depthPatch = single(depthPatch);
    
%     % Code to visualize TDF voxel grid
%     [gridX,gridY,gridZ] = ndgrid(1:size(voxelGridTDF,1),1:size(voxelGridTDF,2),1:size(voxelGridTDF,3));
%     volPts = [gridX(:),gridY(:),gridZ(:)];
%     volPts = volPts(find(voxelGridTDF > 0),:);
%     cmap = jet;
%     colorPts = cmap(round(voxelGridTDF*63)+1,:);
%     colorPts = colorPts(find(voxelGridTDF > 0),:);
%     pcshow(pointCloud(volPts,'color',colorPts));
end

