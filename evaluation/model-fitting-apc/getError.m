% Compute error between predicted 6D poses and ground truth labels of the
% 'Shelf & Tote' benchmark dataset.
%
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the APC Vision Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% User configurations (change me)
benchmarkPath = '/home/andyz/apc/toolbox/data/benchmark'; % Benchmark dataset directory
predictedPosesFile = '3dmatch-apc-predictions.mat';
groundTruthPosesFile = fullfile('testing-split.mat');
objectInfoFile = fullfile(benchmarkPath,'objects.mat');

% Load object poses (predictions and ground truth labels)
tmpFile = load(predictedPosesFile);
predictedPoses = tmpFile.predictions;
tmpFile = load(groundTruthPosesFile);
groundTruthPoses = tmpFile.testLabels;

% Create scene-object string pairs for ground truth data (to make searching easier)
sceneObjectPairs = cell(length(groundTruthPoses),1);
for labelIdx = 1:length(groundTruthPoses)
    tmpEntry = groundTruthPoses{labelIdx};
    sceneObjectPairs{labelIdx} = sprintf('%s %s',tmpEntry.sceneName,tmpEntry.objectName);
end

% Get object properties
tmpFile = load(objectInfoFile);
objectInfo = tmpFile.objects;
load('objectNames.mat');
        
% For each predicted object pose label, compare to ground truth and show
% accuracy per subset of benchmark dataset
% 1 - all scenes
% 2 - scenes from finals during competition
% 3 - scenes in office environment
% 4 - scenes in warehouse environment
% 5 - scenes of the shelf
% 6 - scenes of the tote
% 7 - scenes with clutter level: 1 - 3 objects
% 8 - scenes with clutter level: 4 - 5 objects
% 9 - scenes with clutter level: 6+ objects
% 10 - objects with < X% occluded objects
% 11 - objects with < X% occluded objects
% 12 - objects with > X% occluded objects
% 13 - objects that are deformable
% 14 - objects with missing depth from sensor
% 15 - objects that are thin
subsetAcc = {};
subsetSegmPrecRec = {};
for evalIdx = 1:15
    subsetAcc{evalIdx} = [];
    subsetSegmPrecRec{evalIdx} = [];
end

for labelIdx = 1:length(predictedPoses)
    tmpEntry = predictedPoses{labelIdx};
    objIdx = find(~cellfun(@isempty,strfind(objectNames,tmpEntry.objectName)));
    
    % Check if there is a ground truth object label in the benchmark
    labelListIdx = find(~cellfun(@isempty,strfind(sceneObjectPairs,sprintf('%s %s',tmpEntry.sceneName,tmpEntry.objectName))));
    labelExists = ~isempty(labelListIdx);
    if labelExists
        
        % Gather corresponding ground truth object labels
        corresGroundTruthPosesWorld = {};
        corresGroundTruthOcclusion = {};
        for instanceIdx = 1:length(labelListIdx)
            corresGroundTruthPosesWorld{instanceIdx} = groundTruthPoses{labelListIdx(instanceIdx)}.objectPose;
            corresGroundTruthOcclusion{instanceIdx} = groundTruthPoses{labelListIdx(instanceIdx)}.occlusion;
        end
        
        currObjPredPose = tmpEntry.objectPose;
        
        if max(isnan(currObjPredPose(:))) % no pose
            continue;
        end
        
        % In the event of multiple instances of the same object, find the closest ground truth label to compare to
        tmpErrorTrans = [];
        closestGroundTruthObjIdx = 1;
        for groundTruthObjIdx = 1:length(corresGroundTruthPosesWorld)
            currObjGroundTruthPose = corresGroundTruthPosesWorld{groundTruthObjIdx};
            tmpErrorTrans = [tmpErrorTrans,sqrt(sum((currObjPredPose(1:3,4)-currObjGroundTruthPose(1:3,4)).^2))];
        end
        [~,closestGroundTruthObjIdx] = min(tmpErrorTrans,[],2);
        currObjGroundTruthPose = corresGroundTruthPosesWorld{closestGroundTruthObjIdx};
        currObjGroundTruthOcclusion = corresGroundTruthOcclusion{closestGroundTruthObjIdx};

        % Compute rotational error
        rotationDiff = inv(currObjPredPose(1:3,1:3))*inv(currObjGroundTruthPose(1:3,1:3))';
        errorRotZYX = abs(rad2deg(rotm2eul(rotationDiff)));
        errorRotXYZ = [errorRotZYX(3),errorRotZYX(2),errorRotZYX(1)];
        symmetryInfo = [objectInfo{objIdx}.xSymmetry,objectInfo{objIdx}.ySymmetry,objectInfo{objIdx}.zSymmetry];
        for dimIdx = 1:3
            if symmetryInfo(dimIdx) == 90 % Square symmetry
                errorRotXYZ(dimIdx) = abs(errorRotXYZ(dimIdx)-90);
                errorRotXYZ(dimIdx) = min(errorRotXYZ(dimIdx),90-errorRotXYZ(dimIdx));
            end
            if symmetryInfo(dimIdx) == 180 % Axis symmetry
                errorRotXYZ(dimIdx) = min(errorRotXYZ(dimIdx),180-errorRotXYZ(dimIdx));
            end
            if symmetryInfo(dimIdx) == 360 % Radial symmetry
                errorRotXYZ(dimIdx) = 0;
            end
        end
        errorRot = mean(errorRotXYZ);

        % Compute translational error
        errorTrans = sqrt(sum((currObjPredPose(1:3,4)-currObjGroundTruthPose(1:3,4)).^2));
  
        % Check if predicted pose is good
        goodRot = errorRot<15;
        goodTrans = errorTrans<0.05;
        
        % Get segmentation precision and recall
        segmPr = tmpEntry.segmPrecision;
        segmRc = tmpEntry.segmRecall;
        
        % If is valid prediction
        if tmpEntry.confidence > 0.0

            % All objects
            subsetAcc{1} = [subsetAcc{1};goodRot,goodTrans];
            subsetSegmPrecRec{1} = [subsetSegmPrecRec{1};segmPr,segmRc];
            
            % If scene is from competition finals
            if ~isempty(strfind(tmpEntry.sceneName,'competition'))
                subsetAcc{2} = [subsetAcc{2};goodRot,goodTrans];
                subsetSegmPrecRec{2} = [subsetSegmPrecRec{2};segmPr,segmRc];
            end

            % If scene is from office/warehouse environment
            if ~isempty(strfind(tmpEntry.sceneName,'office'))
                subsetAcc{3} = [subsetAcc{3};goodRot,goodTrans];
                subsetSegmPrecRec{3} = [subsetSegmPrecRec{3};segmPr,segmRc];
            else
                subsetAcc{4} = [subsetAcc{4};goodRot,goodTrans];
                subsetSegmPrecRec{4} = [subsetSegmPrecRec{4};segmPr,segmRc];
            end

            % If scene is from shelf/tote
            if ~isempty(strfind(tmpEntry.sceneName,'shelf'))
                subsetAcc{5} = [subsetAcc{5};goodRot,goodTrans];
                subsetSegmPrecRec{5} = [subsetSegmPrecRec{5};segmPr,segmRc];
            else
                subsetAcc{6} = [subsetAcc{6};goodRot,goodTrans];
                subsetSegmPrecRec{6} = [subsetSegmPrecRec{6};segmPr,segmRc];
            end

            % If scene is cluttered
            fid = fopen(fullfile(benchmarkPath,tmpEntry.sceneName,'cam.info.txt'),'rb');
            fgetl(fid);
            fgetl(fid);
            objListLine = fgetl(fid);
            objListDelim = strsplit(objListLine,'"');
            sceneObjects = {};
            for tmpObjListIdx = 2:2:length(objListDelim)
                sceneObjects{length(sceneObjects)+1} = objListDelim{tmpObjListIdx};
            end
            fclose(fid);
            if length(sceneObjects) == 1 || length(sceneObjects) == 2 || length(sceneObjects) == 3
                subsetAcc{7} = [subsetAcc{7};goodRot,goodTrans];
                subsetSegmPrecRec{7} = [subsetSegmPrecRec{7};segmPr,segmRc];
            elseif length(sceneObjects) == 4 || length(sceneObjects) == 5
                subsetAcc{8} = [subsetAcc{8};goodRot,goodTrans];
                subsetSegmPrecRec{8} = [subsetSegmPrecRec{8};segmPr,segmRc];
            else
                subsetAcc{9} = [subsetAcc{9};goodRot,goodTrans];
                subsetSegmPrecRec{9} = [subsetSegmPrecRec{9};segmPr,segmRc];
            end

            % If object is occluded
            if nanmean(currObjGroundTruthOcclusion) < 5
                subsetAcc{10} = [subsetAcc{10};goodRot,goodTrans];
                subsetSegmPrecRec{10} = [subsetSegmPrecRec{10};segmPr,segmRc];
            elseif nanmean(currObjGroundTruthOcclusion) < 30
                subsetAcc{11} = [subsetAcc{11};goodRot,goodTrans];
                subsetSegmPrecRec{11} = [subsetSegmPrecRec{11};segmPr,segmRc];
            else
                subsetAcc{12} = [subsetAcc{12};goodRot,goodTrans];
                subsetSegmPrecRec{12} = [subsetSegmPrecRec{12};segmPr,segmRc];
            end

            % If object is deformable
            if objectInfo{objIdx}.isDeformable
                subsetAcc{13} = [subsetAcc{13};goodRot,goodTrans];
                subsetSegmPrecRec{13} = [subsetSegmPrecRec{13};segmPr,segmRc];
            end

            % If object has missing depth
            if objectInfo{objIdx}.noDepth
                subsetAcc{14} = [subsetAcc{14};goodRot,goodTrans];
                subsetSegmPrecRec{14} = [subsetSegmPrecRec{14};segmPr,segmRc];
            end

            % If object is thin
            if objectInfo{objIdx}.isThin
                subsetAcc{15} = [subsetAcc{15};goodRot,goodTrans];
                subsetSegmPrecRec{15} = [subsetSegmPrecRec{15};segmPr,segmRc];
            end
        end
    end
end

% Compute average 6D pose estimation performance and report rotation/translation accuracies
fprintf('Predicted 6D Object Pose Performance (%% rotation/translation accuracy):\n')
fprintf('    Total: \t%.2f / %.2f\n',sum(subsetAcc{1},1)/size(subsetAcc{1},1)*100);
fprintf('    Cptn: \t%.2f / %.2f\n',sum(subsetAcc{2},1)/size(subsetAcc{2},1)*100);
fprintf('    Office: \t%.2f / %.2f\n',sum(subsetAcc{3},1)/size(subsetAcc{3},1)*100);
fprintf('    Warehouse: \t%.2f / %.2f\n',sum(subsetAcc{4},1)/size(subsetAcc{4},1)*100);
fprintf('    Shelf: \t%.2f / %.2f\n',sum(subsetAcc{5},1)/size(subsetAcc{5},1)*100);
fprintf('    Tote: \t%.2f / %.2f\n',sum(subsetAcc{6},1)/size(subsetAcc{6},1)*100);
fprintf('    Cltr 1-3: \t%.2f / %.2f\n',sum(subsetAcc{7},1)/size(subsetAcc{7},1)*100);
fprintf('    Cltr 4-5: \t%.2f / %.2f\n',sum(subsetAcc{8},1)/size(subsetAcc{8},1)*100);
fprintf('    Cltr 6+: \t%.2f / %.2f\n',sum(subsetAcc{9},1)/size(subsetAcc{9},1)*100);
fprintf('    Occ < X%%: \t%.2f / %.2f\n',sum(subsetAcc{10},1)/size(subsetAcc{10},1)*100);
fprintf('    Occ < X%%: \t%.2f / %.2f\n',sum(subsetAcc{11},1)/size(subsetAcc{11},1)*100);
fprintf('    Occ > X%%: \t%.2f / %.2f\n',sum(subsetAcc{12},1)/size(subsetAcc{12},1)*100);
fprintf('    Deform: \t%.2f / %.2f\n',sum(subsetAcc{13},1)/size(subsetAcc{13},1)*100);
fprintf('    No Depth: \t%.2f / %.2f\n',sum(subsetAcc{14},1)/size(subsetAcc{14},1)*100);
fprintf('    Thin: \t%.2f / %.2f\n',sum(subsetAcc{15},1)/size(subsetAcc{15},1)*100);

% Compute average segmentation performance and report precision/recall
for evalIdx = 1:15
    subsetSegmPrecRec{evalIdx} = nanmean(subsetSegmPrecRec{evalIdx});
end
fprintf('2D Object Segmentation Performance (%% precision/recall):\n')
fprintf('    Total: \t%.2f / %.2f\n',subsetSegmPrecRec{1}*100);
fprintf('    Cptn: \t%.2f / %.2f\n',subsetSegmPrecRec{2}*100);
fprintf('    Office: \t%.2f / %.2f\n',subsetSegmPrecRec{3}*100);
fprintf('    Warehouse: \t%.2f / %.2f\n',subsetSegmPrecRec{4}*100);
fprintf('    Shelf: \t%.2f / %.2f\n',subsetSegmPrecRec{5}*100);
fprintf('    Tote: \t%.2f / %.2f\n',subsetSegmPrecRec{6}*100);
fprintf('    Cltr 1-3: \t%.2f / %.2f\n',subsetSegmPrecRec{7}*100);
fprintf('    Cltr 4-5: \t%.2f / %.2f\n',subsetSegmPrecRec{8}*100);
fprintf('    Cltr 6+: \t%.2f / %.2f\n',subsetSegmPrecRec{9}*100);
fprintf('    Occ < X%%: \t%.2f / %.2f\n',subsetSegmPrecRec{10}*100);
fprintf('    Occ < X%%: \t%.2f / %.2f\n',subsetSegmPrecRec{11}*100);
fprintf('    Occ > X%%: \t%.2f / %.2f\n',subsetSegmPrecRec{12}*100);
fprintf('    Deform: \t%.2f / %.2f\n',subsetSegmPrecRec{13}*100);
fprintf('    No Depth: \t%.2f / %.2f\n',subsetSegmPrecRec{14}*100);
fprintf('    Thin: \t%.2f / %.2f\n',subsetSegmPrecRec{15}*100);

% Report segmentation F-scores
fprintf('Segmentation F-scores: %.1f',2*(prod(subsetSegmPrecRec{1})/sum(subsetSegmPrecRec{1}))*100);
for i = 2:15
    fprintf(', %.1f', 2*(prod(subsetSegmPrecRec{i})/sum(subsetSegmPrecRec{i}))*100);
end
fprintf('\n');
