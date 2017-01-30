% Script to run create evaluation .log files from predicted rigid
% transformations for the geometric registration benchmark 
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
descriptorName = '3dmatch';
dataPath = '../../data/fragments'; % Location of scene files
intermPath = '../../data/fragments/intermediate-files'; % Location of intermediate files holding keypoints and descriptor vectors
savePath = '../../data/fragments'; % Location to save evaluation .log file
% % Synthetic data benchmark
% sceneList = {'iclnuim-livingroom1-evaluation' ...
%              'iclnuim-livingroom2-evaluation' ...
%              'iclnuim-office1-evaluation' ...
%              'iclnuim-office2-evaluation'};  
% Real data benchmark
sceneList = {'7-scenes-redkitchen', ...
             'sun3d-home_at-home_at_scan1_2013_jan_1', ...
             'sun3d-home_md-home_md_scan9_2012_sep_30', ...
             'sun3d-hotel_uc-scan3', ...
             'sun3d-hotel_umd-maryland_hotel1', ...
             'sun3d-hotel_umd-maryland_hotel3', ...
             'sun3d-mit_76_studyroom-76-1studyroom2', ...
             'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'};

totalRecall = []; totalPrecision = [];
for sceneIdx = 1:length(sceneList)
    
    % List fragment files
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    sceneDir = dir(fullfile(scenePath,'*.ply'));
    numFragments = length(sceneDir);
    fprintf('%s\n',scenePath);

    % Loop through registration results and write a log file
    logPath = fullfile(savePath,sprintf('%s-evaluation',sceneList{sceneIdx}),sprintf('%s.log',descriptorName));
    fid = fopen(logPath,'w');
    for fragment1Idx = 1:numFragments
        for fragment2Idx = (fragment1Idx+1):numFragments
            fragment1Name = sprintf('cloud_bin_%d',fragment1Idx-1);
            fragment2Name = sprintf('cloud_bin_%d',fragment2Idx-1);

            resultPath = fullfile(intermPath,sceneList{sceneIdx},sprintf('%s-registration-results',descriptorName),sprintf('%s-%s.rt.txt',fragment1Name,fragment2Name));

            resultRt = dlmread(resultPath,'\t',[2,0,5,3]);
            resultNumInliers = dlmread(resultPath,'\t',[1,0,1,0]);
            resultInlierRatio = dlmread(resultPath,'\t',[1,1,1,1]);
            resultAlignRatio = dlmread(resultPath,'\t',[1,2,1,3]);
            
            % Check if surface overlap is above some threshold
            % Note: set threshold to 0.23 to reproduce our numbers 
            % for the synthetic benchmark from Choi et al.
            if resultAlignRatio(1) > 0.30
                fprintf(fid,'%d\t %d\t %d\t\n',fragment1Idx-1,fragment2Idx-1,numFragments);
                fprintf(fid,'%.10f\t%.10f\t%.10f\t%.10f\n',resultRt');
            end
        end
    end
    fclose(fid);
    
end
