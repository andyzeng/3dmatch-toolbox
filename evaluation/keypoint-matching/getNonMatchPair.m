function [p1,p2] = getNonMatchPair(sceneDataList,maxTries,voxelGridPatchRadius,voxelSize,voxelMargin)
%GETMATCHPAIR Summary of this function goes here
%   Detailed explanation goes here

nonMatchFound = false;
while ~nonMatchFound

    % Pick a random scene and a set of random frames
    randSceneIdx = randsample(length(sceneDataList),1);
    randFrameIdx = randsample(length(sceneDataList{randSceneIdx}.frameList),maxTries);
    camK = sceneDataList{randSceneIdx}.camK;

    % Find a random 3D point (in world coordinates) in a random frame
    framePrefix = fullfile(sceneDataList{randSceneIdx}.frameList{randFrameIdx(1)});
    p1 = [];
    p1.framePath = framePrefix;
    depthIm = imread(strcat(framePrefix,'.depth.png'));
    depthIm = double(depthIm)./1000;
    depthIm(find(depthIm > 6)) = 0;
    randDepthInd = randsample(find(depthIm > 0),1);
    [pixY,pixX] = ind2sub(size(depthIm),randDepthInd);
    p1.pixelCoords = [pixX-1,pixY-1];
%     p1.x = pixX;
%     p1.y = pixY;
    ptCamZ = depthIm(randDepthInd);
    ptCamX = (pixX-0.5-camK(1,3))*ptCamZ/camK(1,1);
    ptCamY = (pixY-0.5-camK(2,3))*ptCamZ/camK(2,2);
    ptCam = [ptCamX;ptCamY;ptCamZ];
    p1.camCoords = ptCam;
%     p1.camx = ptCam(1);
%     p1.camy = ptCam(2);
%     p1.camz = ptCam(3);
    extCam2World = dlmread(strcat(framePrefix,'.pose.txt'));
    p1World = extCam2World(1:3,1:3)*ptCam + extCam2World(1:3,4);

    % Compute bounding box in pixel coordinates
    bboxRange = [ptCam(1)-voxelGridPatchRadius*voxelSize,ptCam(1)+voxelGridPatchRadius*voxelSize; ...
                 ptCam(2)-voxelGridPatchRadius*voxelSize,ptCam(2)+voxelGridPatchRadius*voxelSize; ...
                 ptCam(3)-voxelGridPatchRadius*voxelSize,ptCam(3)+voxelGridPatchRadius*voxelSize];
    bboxCorners = [bboxRange(1,1),bboxRange(1,1),bboxRange(1,1),bboxRange(1,1),bboxRange(1,2),bboxRange(1,2),bboxRange(1,2),bboxRange(1,2); ...
                   bboxRange(2,1),bboxRange(2,1),bboxRange(2,2),bboxRange(2,2),bboxRange(2,1),bboxRange(2,1),bboxRange(2,2),bboxRange(2,2); ...
                   bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2)];
    p1.bboxCornersCam = bboxCorners;
    bboxPixX = round((bboxCorners(1,:).*camK(1,1)./bboxCorners(3,:))+camK(1,3));
    bboxPixY = round((bboxCorners(2,:).*camK(2,2)./bboxCorners(3,:))+camK(2,3));
    bboxPixX = [pixX-max([pixX-bboxPixX,bboxPixX-pixX]),pixX+max([pixX-bboxPixX,bboxPixX-pixX])];
    bboxPixY = [pixY-max([pixY-bboxPixY,bboxPixY-pixY]),pixY+max([pixY-bboxPixY,bboxPixY-pixY])];
    if sum(bboxPixX <= 0 | bboxPixX > 640 | bboxPixY <= 0 | bboxPixY > 480) > 0
        continue;
    end
    p1.bboxRangePixels = [bboxPixX;bboxPixY];
    p1.camK = camK;
    
    % Find another random 3D point (in world coordinates) in a random frame
    for otherFrameIdx = 2:length(randFrameIdx)
        otherFramePrefix = fullfile(sceneDataList{randSceneIdx}.frameList{randFrameIdx(otherFrameIdx)});
        p2 = [];
        p2.framePath = otherFramePrefix;
        depthIm = imread(strcat(otherFramePrefix,'.depth.png'));
        depthIm = double(depthIm)./1000;
        depthIm(find(depthIm > 6)) = 0;
        randDepthInd = randsample(find(depthIm > 0),1);
        [pixY,pixX] = ind2sub(size(depthIm),randDepthInd);
        p2.pixelCoords = [pixX-1,pixY-1];
%         p2.x = pixX;
%         p2.y = pixY;
        ptCamZ = depthIm(randDepthInd);
        ptCamX = (pixX-0.5-camK(1,3))*ptCamZ/camK(1,1);
        ptCamY = (pixY-0.5-camK(2,3))*ptCamZ/camK(2,2);
        ptCam = [ptCamX;ptCamY;ptCamZ];
        p2.camCoords = ptCam;
%         p2.camx = ptCam(1);
%         p2.camy = ptCam(2);
%         p2.camz = ptCam(3);
        extCam2World = dlmread(strcat(otherFramePrefix,'.pose.txt'));
        p2World = extCam2World(1:3,1:3)*ptCam + extCam2World(1:3,4);

        % Compute bounding box in pixel coordinates
        bboxRange = [ptCam(1)-voxelGridPatchRadius*voxelSize,ptCam(1)+voxelGridPatchRadius*voxelSize; ...
                     ptCam(2)-voxelGridPatchRadius*voxelSize,ptCam(2)+voxelGridPatchRadius*voxelSize; ...
                     ptCam(3)-voxelGridPatchRadius*voxelSize,ptCam(3)+voxelGridPatchRadius*voxelSize];
        bboxCorners = [bboxRange(1,1),bboxRange(1,1),bboxRange(1,1),bboxRange(1,1),bboxRange(1,2),bboxRange(1,2),bboxRange(1,2),bboxRange(1,2); ...
                       bboxRange(2,1),bboxRange(2,1),bboxRange(2,2),bboxRange(2,2),bboxRange(2,1),bboxRange(2,1),bboxRange(2,2),bboxRange(2,2); ...
                       bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2)];
        p2.bboxCornersCam = bboxCorners;
        bboxPixX = round((bboxCorners(1,:).*camK(1,1)./bboxCorners(3,:))+camK(1,3));
        bboxPixY = round((bboxCorners(2,:).*camK(2,2)./bboxCorners(3,:))+camK(2,3));
        bboxPixX = [pixX-max([pixX-bboxPixX,bboxPixX-pixX]),pixX+max([pixX-bboxPixX,bboxPixX-pixX])];
        bboxPixY = [pixY-max([pixY-bboxPixY,bboxPixY-pixY]),pixY+max([pixY-bboxPixY,bboxPixY-pixY])];
        if sum(bboxPixX <= 0 | bboxPixX > 640 | bboxPixY <= 0 | bboxPixY > 480) > 0
            continue;
        end
        p2.bboxRangePixels = [bboxPixX;bboxPixY];
        p2.camK = camK;
        
        % Check if 3D point is far enough to be considered a nonmatch
        if sqrt(sum((p1World-p2World).^2)) > 0.1
            nonMatchFound = true;
            break;
        end
    
    end
    
end

% Get color/depth image patches and local TDF voxel grids
[p1.colorPatch,p1.depthPatch,p1.voxelGridTDF] = getPatchData(p1,voxelGridPatchRadius,voxelSize,voxelMargin);
[p2.colorPatch,p2.depthPatch,p2.voxelGridTDF] = getPatchData(p2,voxelGridPatchRadius,voxelSize,voxelMargin);

end