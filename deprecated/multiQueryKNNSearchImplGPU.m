function [indices,dists] = multiQueryKNNSearchImplGPU(ptCloudB,locA)
% GPU version of Matlab's multiQueryKNNSearchImpl for K = 1

global KNNSearchGPU

numBlocks = ceil(size(locA,1)/512);
numThreads = 512;
KNNSearchGPU.GridSize = [numBlocks 1];
KNNSearchGPU.ThreadBlockSize = [numThreads 1];
resultsGPU = gpuArray(single(zeros(1,2*size(locA,1))));
argsGPU = gpuArray(int32([numBlocks numThreads size(ptCloudB.Location,1) size(locA,1)]));
pcGPU1 = gpuArray(single([ptCloudB.Location(:,1)' ptCloudB.Location(:,2)' ptCloudB.Location(:,3)']));
pcGPU2 = gpuArray(single([locA(:,1)' locA(:,2)' locA(:,3)']));
results = feval(KNNSearchGPU,resultsGPU,argsGPU,pcGPU1,pcGPU2);
indices = gather(results(1:size(locA,1)));
dists = gather(results((size(locA,1)+1):end));

end

