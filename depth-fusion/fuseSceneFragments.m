% Script to fuse depth maps from real-world test scenes to generate scene 
% fragments. Fragments are used as part of the real-world geometric
% registration benchmark, in the same spirit as Choi et al 2015. See our
% paper for more details.
% 
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% User configurations (change me)
cudaBinPath = '/usr/local/cuda/bin';
cudaLibPath = '/usr/local/cuda/lib64';
cudnnLibPath = '/usr/local/cudnn/v5.1/lib64';
numFramesPerFrag = 50;
voxelSize = 0.006;
truncMargin = voxelSize*5;
voxelGridOrigin = [-1.5,-1.5,0.5];

dataPath = '../data/test';
sceneList = {'7-scenes-redkitchen', ...
             'sun3d-hotel_umd-maryland_hotel3', ...
             'sun3d-mit_76_studyroom-76-1studyroom2', ...
             'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika', ...
             'sun3d-home_at-home_at_scan1_2013_jan_1', ...
             'sun3d-home_md-home_md_scan9_2012_sep_30', ...
             'sun3d-hotel_uc-scan3', ...
             'sun3d-hotel_umd-maryland_hotel1'};
         
% ./demo ../data/sample/depth-fusion-demo/camera-intrinsics.txt ../data/sample/depth-fusion-demo/rgbd-frames 8 8 50 -1.5 -1.5 0.5 0.006 0.03

for sceneIdx = 1:length(sceneList)

    sceneName = sceneList{sceneIdx};

    scenePath = fullfile(dataPath,sceneName);
    seqList = dir(fullfile(scenePath,'seq-*'));
    
    camKPath = fullfile(scenePath,'camera-intrinsics.txt');

    cloudIdx = 0;
    
    mkdir(fullfile(fragmentsPath,sceneName));

    for seqIdx = 1:min(length(seqList),3)

        seqName = seqList(seqIdx).name; 
        seqPath = fullfile(scenePath,seqName);
        frameList = dir(fullfile(seqPath,'*.depth.png'));
        
        % frameIdx = 0:numFramesPerFrag:(length(frameList)-1-numFramesPerFrag) % all other scenes
        % frameIdx = 0:100:(6000-1-numFramesPerFrag) % sun3d-home_at-home_at_scan1_2013_jan_1, sun3d-home_md-home_md_scan9_2012_sep_30
        % frameIdx = [0:100:(2750-1),7500:100:(length(frameList)-numFramesPerFrag-1)] % sun3d-hotel_uc-scan3
        % 0:100:(length(frameList)-1-numFramesPerFrag) % sun3d-hotel_umd-maryland_hotel1
        for frameIdx = 0:numFramesPerFrag:(length(frameList)-1-numFramesPerFrag)
            
            % System call to fuse depth maps into a scene fragment
            system(sprintf('export PATH=$PATH:%s',cudaBinPath));
            system(sprintf('unset LD_LIBRARY_PATH; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:%s:%s; ./demo %s %s %d %d %f %f %f %f %f %f',cudaLibPath,cudnnLibPath,camKPath,seqPath,frameIdx,frameIdx,numFramesPerFrag, ...
                                                                                                                                                                voxelGridOrigin(1),voxelGridOrigin(2),voxelGridOrigin(3),voxelSize,truncMargin));                              
            % Copy point cloud file to fragment folder
            pointCloudFile = fullfile(fragmentsPath,sceneName,sprintf('cloud_bin_%d.ply',cloudIdx));
            movefile('tsdf.ply',pointCloudFile);
            
            % Save camera pose of base frame
            extCam2World = dlmread(fullfile(seqPath,sprintf('frame-%06d.pose.txt',frameIdx)));
            fid = fopen(fullfile(fragmentsPath,sceneName,sprintf('cloud_bin_%d.info.txt',cloudIdx)),'w');
            fprintf(fid,'%s\t %s\t %d\t %d\t\n',sceneName,seqName,frameIdx,frameIdx+49);
            fprintf(fid,'%15.8e\t %15.8e\t %15.8e\t %15.8e\t\n',extCam2World');
            fclose(fid);
            
            cloudIdx = cloudIdx + 1;
        end

    end

end

