% Do system calls to get3DMatchDescriptors.cu to compute 3DMatch
% descriptors for sets of keypoints from object models and segmentation
% point clouds
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
dataPath = '../../data/apc';
descriptorName = '3dmatch';
cudaBinPath = '/usr/local/cuda/bin';
cudaLibPath = '/usr/local/cuda/lib64';
cudnnLibPath = '/usr/local/cudnn/v5.1/lib64';

% Compute keypoint descriptors for segmentation point clouds
scenarioDir = dir(fullfile(dataPath,'scenarios/00*'));
for scenarioIdx = 0:(length(scenarioDir)-1)
    scenarioPath = fullfile(dataPath,'scenarios',sprintf('%06d',scenarioIdx));
    fprintf('%s\n',scenarioPath);
    
    % Load segmentation point cloud and TDF voxel grid
    segmPointCloud = pcread(fullfile(scenarioPath,'segmentation.ply'));
    segmVoxelGridFile = fullfile(scenarioPath,'segmentation.TDF.bin');
    fileID = fopen(segmVoxelGridFile,'rb');
    segmVoxelGridSize = fread(fileID,3,'single');
    segmVoxelGridOrigin = fread(fileID,3,'single');
    segm2modelGroundTruth = reshape(fread(fileID,16,'single'),4,4)';
    segmVoxelGridData = fread(fileID,'single');
    fclose(fileID);
    segmVoxelGrid = reshape(segmVoxelGridData,segmVoxelGridSize');

    delete('data.bin');
    delete('feat.bin');

    % Load keypoints
    keypointsFile = fullfile(scenarioPath,'segmentation.keypoints.bin');
    fileID = fopen(keypointsFile,'rb');
    numKeypoints = fread(fileID,1,'single');
    keypointsCamData = fread(fileID,'single');
    fclose(fileID);
    keypointsCam = reshape(keypointsCamData,3,numKeypoints)';
    keypointsGrid = round((keypointsCam-repmat(segmVoxelGridOrigin',numKeypoints,1))./0.005)+1;

    % Write keypoints to data.bin
    fileID = fopen('data.bin','wb');
    fwrite(fileID,numKeypoints,'single');
    fwrite(fileID,[30,30,30],'single');
    for keypointIdx = 1:numKeypoints
        currKeypoint = keypointsGrid(keypointIdx,:);
        currLocalVoxelGrid = segmVoxelGrid((currKeypoint(1)-14):(currKeypoint(1)+15),(currKeypoint(2)-14):(currKeypoint(2)+15),(currKeypoint(3)-14):(currKeypoint(3)+15));
        fwrite(fileID,currLocalVoxelGrid,'single');
    end
    fclose(fileID);

    % Modify 3DMatch architecture to support current voxel grid size
    fileID1 = fopen('test.json','r');
    fildID2 = fopen('tmp.json','w');
    while(~feof(fileID1))
       lineStr = fgetl(fileID1);
       lineStr = strrep(lineStr,'dataBatchSize',sprintf('%d',numKeypoints));
       lineStr = strrep(lineStr,'dataSizeX',sprintf('%d',30));
       lineStr = strrep(lineStr,'dataSizeY',sprintf('%d',30));
       lineStr = strrep(lineStr,'dataSizeZ',sprintf('%d',30));
       fprintf(fildID2,'%s\n',lineStr);
    end
    fclose(fileID1);
    fclose(fildID2);

    % Run Marvin to extract 3DMatch descriptors
    system(sprintf('export PATH=$PATH:%s',cudaBinPath));
    system(sprintf('unset LD_LIBRARY_PATH; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:%s:%s; ./get3DMatchDescriptors %s',cudaLibPath,cudnnLibPath,'data.bin'));

    % Load descriptors of keypoints near surface
    fileID = fopen('feat.bin','rb');
    descriptorDataSize = fread(fileID,5,'single');
    descriptorSize = 512;
    descriptorData = fread(fileID,prod(descriptorDataSize),'single');
    fclose(fileID);
    keypointDescriptors = reshape(descriptorData,descriptorSize,numKeypoints)';
    
    % Save keypoint descriptors
    fileID = fopen(fullfile(scenarioPath,sprintf('segmentation.keypoints.%s.descriptors.bin',descriptorName)),'wb');
    fwrite(fileID,numKeypoints,'single');
    fwrite(fileID,descriptorSize,'single');
    segmKeypointDescriptorsFlipped = keypointDescriptors';
    fwrite(fileID,segmKeypointDescriptorsFlipped(:),'single');
    fclose(fileID);
end

% Compute keypoint descriptors for object models
objectFiles = dir(fullfile(dataPath,'objects/*.ply'));
for objectIdx = 1:length(objectFiles)
    objectName = objectFiles(objectIdx).name(1:(end-4));
    fprintf('%s\n',objectName);

    % Read object model point cloud and TDF voxel grid
    modelPointCloud = pcread(fullfile(dataPath,'objects',sprintf('%s.ply',objectName)));
    modelVoxelGridFile = fullfile(dataPath,'objects',sprintf('%s.TDF.bin',objectName));
    fileID = fopen(modelVoxelGridFile,'rb');
    modelVoxelGridSize = fread(fileID,3,'single');
    modelVoxelGridOrigin = fread(fileID,3,'single');
    modelVoxelGridData = fread(fileID,'single');
    fclose(fileID);
    modelVoxelGrid = reshape(modelVoxelGridData,modelVoxelGridSize');
    
    delete('data.bin');
    delete('feat.bin');

    % Load keypoints
    keypointsFile = fullfile(dataPath,'objects',sprintf('%s.keypoints.bin',objectName));
    fileID = fopen(keypointsFile,'rb');
    numKeypoints = fread(fileID,1,'single');
    keypointsCamData = fread(fileID,'single');
    fclose(fileID);
    keypointsCam = reshape(keypointsCamData,3,numKeypoints)';
    keypointsGrid = round((keypointsCam-repmat(modelVoxelGridOrigin',numKeypoints,1))./0.005)+1;

    % Write keypoints to data.bin
    fileID = fopen('data.bin','wb');
    fwrite(fileID,numKeypoints,'single');
    fwrite(fileID,[30,30,30],'single');
    for keypointIdx = 1:numKeypoints
        currKeypoint = keypointsGrid(keypointIdx,:);
        currLocalVoxelGrid = modelVoxelGrid((currKeypoint(1)-14):(currKeypoint(1)+15),(currKeypoint(2)-14):(currKeypoint(2)+15),(currKeypoint(3)-14):(currKeypoint(3)+15));
        fwrite(fileID,currLocalVoxelGrid,'single');
    end
    fclose(fileID);

    % Modify 3DMatch architecture to support current voxel grid size
    fileID1 = fopen('test.json','r');
    fildID2 = fopen('tmp.json','w');
    while(~feof(fileID1))
       lineStr = fgetl(fileID1);
       lineStr = strrep(lineStr,'dataBatchSize',sprintf('%d',numKeypoints));
       lineStr = strrep(lineStr,'dataSizeX',sprintf('%d',30));
       lineStr = strrep(lineStr,'dataSizeY',sprintf('%d',30));
       lineStr = strrep(lineStr,'dataSizeZ',sprintf('%d',30));
       fprintf(fildID2,'%s\n',lineStr);
    end
    fclose(fileID1);
    fclose(fildID2);

    % Run Marvin to extract 3DMatch descriptors
    system(sprintf('export PATH=$PATH:%s',cudaBinPath));
    system(sprintf('unset LD_LIBRARY_PATH; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:%s:%s; ./get3DMatchDescriptors %s',cudaLibPath,cudnnLibPath,'data.bin'));
    
    % Load descriptors of keypoints near surface
    fileID = fopen('feat.bin','rb');
    descriptorDataSize = fread(fileID,5,'single');
    descriptorSize = 512;
    descriptorData = fread(fileID,prod(descriptorDataSize),'single');
    fclose(fileID);
    keypointDescriptors = reshape(descriptorData,descriptorSize,numKeypoints)';
    
    % Save keypoint descriptors
    fileID = fopen(fullfile(dataPath,'objects',sprintf('%s.keypoints.%s.descriptors.bin',objectName,descriptorName)),'wb');
    fwrite(fileID,numKeypoints,'single');
    fwrite(fileID,descriptorSize,'single');
    modelKeypointDescriptorsFlipped = keypointDescriptors';
    fwrite(fileID,modelKeypointDescriptorsFlipped(:),'single');
    fclose(fileID);
end