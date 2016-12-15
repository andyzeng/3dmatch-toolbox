% Run 3DMatch on the validation set for keypoint matching and write a
% corresponding .log file for evaluation.

% CUDA/CUDNN paths (for system call)
cudaBinPath = '/usr/local/cuda/bin';
cudaLibPath = '/usr/local/cuda/lib64';
cudnnLibPath = '/usr/local/cudnn/v5.1/lib64';

% Load validation set data
fprintf('Loading test data ...\n'); tic;
load('test-set.mat');
toc;
numComparisons = size(data,1);
numKeypoints = numComparisons*2;

% Write keypoints to data.bin
fileID = fopen('data.bin','wb');
fwrite(fileID,numKeypoints,'single');
fwrite(fileID,[30,30,30],'single');
for keypointIdx = 1:numKeypoints
    [pointIdx,comparisonIdx] = ind2sub(size(data'),keypointIdx);
    localVoxelGridTDF = data{comparisonIdx,pointIdx}.voxelGridTDF;
    fwrite(fileID,localVoxelGridTDF,'single');
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

% Load 3DMatch descriptors of keypoints
fileID = fopen('feat.bin','rb');
descDataSize = fread(fileID,5,'single');
descSize = 512;
descData = fread(fileID,'single');
fclose(fileID);
keyptDesc = reshape(descData,descSize,numKeypoints)';

% Compute descriptor distances for each keypoint comparison
descDists = zeros(numComparisons,1);
for comparisonIdx = 1:numComparisons
    p1Descriptor = keyptDesc((comparisonIdx-1)*2+1,:);
    p2Descriptor = keyptDesc((comparisonIdx-1)*2+2,:);
    descDists(comparisonIdx) = sqrt(sum((p1Descriptor-p2Descriptor).^2));
end

% Save descriptor distances to a log file
dlmwrite('3dmatch.log',[numComparisons;descDists]);
