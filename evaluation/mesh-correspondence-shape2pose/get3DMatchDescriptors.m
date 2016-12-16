% Given surface keypoint locations and a TDF voxel grid volume of the
% object, compute 3DMatch descriptors for all keypoints
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
dataPath = '../../data/shape2pose/carts';

objectFiles = dir(fullfile(dataPath,'*.raw'));
for objectIdx = 1:length(objectFiles)
    objectPrefix = objectFiles(objectIdx).name(1:(end-4));
    fprintf('%s\n',objectPrefix);
    keypointsFile = fullfile(dataPath,sprintf('%s.keypoints.bin',objectPrefix));
    TDFVoxelGridFile = fullfile(dataPath,sprintf('%s.TDF.bin',objectPrefix));

    % Do system call to C++/CUDA code to compute 3DMatch descriptors
    system('export PATH=$PATH:/usr/local/cuda/bin');
    system(sprintf('unset LD_LIBRARY_PATH; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cudnn/v5.1/lib64; ./get3DMatchDescriptors %s %s',TDFVoxelGridFile,keypointsFile));

end
