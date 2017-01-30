% Convert benchmark files into a set of scenarios, where each scenario is
% an alignment between pre-scanned object model and segmentation result
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
scenarioDataPath = '../../data/apc/scenarios'; % Path to scenario data

load(fullfile(benchmarkPath,'scenes.mat'));
scenarioIdx = 0;
for sceneIdx = 1:length(scenes)
    tmpScene = scenes{sceneIdx};
    fprintf('%s\n',tmpScene);
    tmpPath = fullfile(sceneDataPath,tmpScene);
    segmDir = dir(fullfile(tmpPath,'*.segm.ply'));
    
    for segmIdx = 1:length(segmDir)
        segmName = segmDir(segmIdx).name(1:(end-9));
        if exist(fullfile(tmpPath,strcat(segmName,'.segmentation.TDF.bin')),'file')
            saveToDir = fullfile(scenarioDataPath,sprintf('%06d',scenarioIdx));
            scenarioIdx = scenarioIdx + 1;
            mkdir(saveToDir);
            copyfile(fullfile(tmpPath,strcat(segmName,'.segm.ply')),fullfile(saveToDir,'segmentation.ply'));
            try
                copyfile(fullfile(tmpPath,strcat(segmName,'.segmentation.TDF.bin')),fullfile(saveToDir,'segmentation.TDF.bin'));
            catch
            end
            objName = segmName(1:(end-2));
            instanceIdx = str2num(segmName(end));
            fid = fopen(fullfile(saveToDir,'info.txt'),'w');
            fprintf(fid,'scene: %s\nobject: %s\ninstance: %d',tmpScene,objName,instanceIdx);
            fclose(fid);
        end
    end
end














