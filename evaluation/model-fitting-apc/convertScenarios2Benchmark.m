% Convert set of scenarios back into the organization of benchmark files,
% and copy over all pose prediction files
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
scenarioDataPath = '../../data/apc/scenarios';
sceneDataPath = '../../data/apc/scenes';
descriptorName = '3dmatch';

scenarioList = dir(fullfile(scenarioDataPath,'00*'));
for scenarioIdx = 0:(length(scenarioList)-1)
    fprintf('%d/%d\n',scenarioIdx,(length(scenarioList)-1));
    scenarioPath = fullfile(scenarioDataPath,sprintf('%06d',scenarioIdx));
    fid = fopen(fullfile(scenarioPath,'info.txt'),'r');
    sceneName = fscanf(fid,'scene: %s\n');
    objName = fscanf(fid,'object: %s\n');
    instanceIdx = fscanf(fid,'instance: %d\n');
    fclose(fid);
    model2segm = dlmread(fullfile(scenarioPath,sprintf('ransac.%s.model2segm.txt',descriptorName)));
    fid = fopen(fullfile(sceneDataPath,sceneName,sprintf('%s.%d.ransac.%s.model2segm.txt',objName,instanceIdx,descriptorName)),'w');
    for i = 1:4
        fprintf(fid,'%15.8e\t',model2segm(i,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
end









