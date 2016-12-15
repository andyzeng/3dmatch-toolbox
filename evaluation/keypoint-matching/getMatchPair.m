function [p1,p2] = getMatchPair(sceneDataList,maxTries,voxelGridPatchRadius,voxelSize,voxelMargin)
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

corresFound = false;
while ~corresFound
    
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
    ptCamZ = depthIm(randDepthInd);
    ptCamX = (pixX-0.5-camK(1,3))*ptCamZ/camK(1,1);
    ptCamY = (pixY-0.5-camK(2,3))*ptCamZ/camK(2,2);
    ptCam = [ptCamX;ptCamY;ptCamZ];
    p1.camCoords = ptCam;
    extCam2World = dlmread(strcat(framePrefix,'.pose.txt'));
    p1CamLoc = extCam2World(1:3,4);
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

%         % Visualize bounding box corners
%         figure(); imagesc(depthIm); hold on; scatter(p1.x,p1.y,'g','fill'); hold off;
%         for i=1:8
%             hold on; scatter(p1.bboxPix(1,i),p1.bboxPix(2,i),'r','fill'); hold off;
%         end

    % Loop through other random frames to find a positive point match
    for otherFrameIdx = 2:length(randFrameIdx)
        otherFramePrefix = fullfile(sceneDataList{randSceneIdx}.frameList{randFrameIdx(otherFrameIdx)});

        % Check if 3D point is within camera view frustum
        extCam2World = dlmread(strcat(otherFramePrefix,'.pose.txt'));
        if isnan(sum(extCam2World(:)))
            continue;
        end
        p2CamLoc = extCam2World(1:3,4);
        if sqrt(sum((p1CamLoc-p2CamLoc).^2)) < 1
            continue;
        end
        extWorld2Cam = inv(extCam2World);
        ptCam = extWorld2Cam(1:3,1:3)*p1World + extWorld2Cam(1:3,4);
        pixX = round((ptCam(1)*camK(1,1)/ptCam(3))+camK(1,3)+0.5);
        pixY = round((ptCam(2)*camK(2,2)/ptCam(3))+camK(2,3)+0.5);
        if pixX > 0 && pixX <= 640 && pixY > 0 && pixY <= 480
            depthIm = imread(strcat(otherFramePrefix,'.depth.png'));
            depthIm = double(depthIm)./1000;
            depthIm(find(depthIm > 6)) = 0;
            if abs(depthIm(pixY,pixX)-ptCam(3)) < 0.03
                ptCamZ = depthIm(pixY,pixX);
                ptCamX = (pixX-0.5-camK(1,3))*ptCamZ/camK(1,1);
                ptCamY = (pixY-0.5-camK(2,3))*ptCamZ/camK(2,2);
                ptCam = [ptCamX;ptCamY;ptCamZ];

                % Compute bounding box in pixel coordinates
                bboxRange = [ptCam(1)-voxelGridPatchRadius*voxelSize,ptCam(1)+voxelGridPatchRadius*voxelSize; ...
                             ptCam(2)-voxelGridPatchRadius*voxelSize,ptCam(2)+voxelGridPatchRadius*voxelSize; ...
                             ptCam(3)-voxelGridPatchRadius*voxelSize,ptCam(3)+voxelGridPatchRadius*voxelSize];
                bboxCorners = [bboxRange(1,1),bboxRange(1,1),bboxRange(1,1),bboxRange(1,1),bboxRange(1,2),bboxRange(1,2),bboxRange(1,2),bboxRange(1,2); ...
                               bboxRange(2,1),bboxRange(2,1),bboxRange(2,2),bboxRange(2,2),bboxRange(2,1),bboxRange(2,1),bboxRange(2,2),bboxRange(2,2); ...
                               bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2),bboxRange(3,1),bboxRange(3,2)];
                p2 = [];
                p2.bboxCornersCam = bboxCorners;
                bboxPixX = round((bboxCorners(1,:).*camK(1,1)./bboxCorners(3,:))+camK(1,3));
                bboxPixY = round((bboxCorners(2,:).*camK(2,2)./bboxCorners(3,:))+camK(2,3));
                bboxPixX = [pixX-max([pixX-bboxPixX,bboxPixX-pixX]),pixX+max([pixX-bboxPixX,bboxPixX-pixX])];
                bboxPixY = [pixY-max([pixY-bboxPixY,bboxPixY-pixY]),pixY+max([pixY-bboxPixY,bboxPixY-pixY])];
                
                if sum(bboxPixX <= 0 | bboxPixX > 640 | bboxPixY <= 0 | bboxPixY > 480) > 0
                    continue;
                end
                p2.bboxRangePixels = [bboxPixX;bboxPixY];

                p2.framePath = otherFramePrefix;
                p2.pixelCoords = [pixX-1,pixY-1];
                p2.camCoords = ptCam;
                p2.camK = camK;
                corresFound = true;
                break;
            end
        end
    end
end

% Get color/depth image patches and local TDF voxel grids
[p1.colorPatch,p1.depthPatch,p1.voxelGridTDF] = getPatchData(p1,voxelGridPatchRadius,voxelSize,voxelMargin);
[p2.colorPatch,p2.depthPatch,p2.voxelGridTDF] = getPatchData(p2,voxelGridPatchRadius,voxelSize,voxelMargin);

end

