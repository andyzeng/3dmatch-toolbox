% Script to load scene fragment point clouds and generate TDF voxel
% volumes, keypoints, and 3DMatch descriptors (intermediate data).
% NOTE: uncomment the code at the end of core/demo.cu to save tdf.bin
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
cudaBinPath = '/usr/local/cuda/bin';
cudaLibPath = '/usr/local/cuda/lib64';
cudnnLibPath = '/usr/local/cudnn/v5.1/lib64';
sceneList = {'7-scenes-redkitchen', ...
             'sun3d-hotel_umd-maryland_hotel3', ...
             'sun3d-mit_76_studyroom-76-1studyroom2', ...
             'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika', ...
             'sun3d-home_at-home_at_scan1_2013_jan_1', ...
             'sun3d-home_md-home_md_scan9_2012_sep_30', ...
             'sun3d-hotel_uc-scan3', ...
             'sun3d-hotel_umd-maryland_hotel1'};
dataPath = '/n/fs/sun3d/andyz/3dmatch/data/fragments';
savePath = '/n/fs/sun3d/andyz/3dmatch/data/fragments/intermediate-files';
         
% Loop through each fragment for all scenes
for sceneIdx = 1:length(sceneList)
    
    sceneName = sceneList{sceneIdx};
    cloudFiles = dir(fullfile(dataPath,sceneName,'cloud_bin*.ply'));
    for cloudIdx = 1:length(cloudFiles)
        cloudPrefix = cloudFiles(cloudIdx).name(1:(end-4));
        fprintf('%s\n',fullfile(dataPath,sceneName,cloudPrefix));
        cloudName = fullfile(dataPath,sceneName,cloudFiles(cloudIdx).name);
        
        % System call to C++/CUDA code to compute intermediate data
        returnPath = pwd;
        cd ../../core;
        system(sprintf('export PATH=$PATH:%s',cudaBinPath));
        system(sprintf('unset LD_LIBRARY_PATH; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:%s:%s; ./demo %s %s',cudaLibPath,cudnnLibPath,cloudName,'test'));

        % Copy files
        fileTDF = fullfile(savePath,sceneName,sprintf('%s.tdf.bin',cloudPrefix));
        fileKeypts = fullfile(savePath,sceneName,sprintf('%s.keypts.bin',cloudPrefix));
        fileDesc = fullfile(savePath,sceneName,sprintf('%s.desc.3dmatch.bin',cloudPrefix));
        copyfile('tdf.bin',fileTDF);
        copyfile('test.keypts.bin',fileKeypts);
        copyfile('test.desc.3dmatch.bin',fileDesc);
        
        cd(returnPath);
    end
end















