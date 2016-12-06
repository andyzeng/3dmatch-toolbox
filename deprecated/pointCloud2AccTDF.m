% Old Matlab-based function to compute TDF voxel grid (more accurate, but significantly slower)
% For this function, compile KNNSearch.cu by running in terminal `nvcc -ptx KNNSearch.cu`
function [voxelGridPts,voxelGridTDF,voxelGridOrigin] = pointCloud2AccTDF(targetPointCloud,voxelSize,truncMargin)

    % Downsample point cloud (to make things slightly faster)
    targetPointCloud = pcdownsample(targetPointCloud,'gridAverage',voxelSize);

    % Compute center coordinates for each voxel of the voxel grid
    [voxelGridPtsX,voxelGridPtsY,voxelGridPtsZ] = ndgrid((targetPointCloud.XLimits(1)-voxelSize*15):voxelSize:(targetPointCloud.XLimits(2)+voxelSize*15), ...
                                                         (targetPointCloud.YLimits(1)-voxelSize*15):voxelSize:(targetPointCloud.YLimits(2)+voxelSize*15), ...
                                                         (targetPointCloud.ZLimits(1)-voxelSize*15):voxelSize:(targetPointCloud.ZLimits(2)+voxelSize*15));
    voxelGridPts = [voxelGridPtsX(:),voxelGridPtsY(:),voxelGridPtsZ(:)];

    % For each voxel, find distance to nearest point on surface point cloud
    % Use GPU kernel function
    global KNNSearchGPU;
    KNNSearchGPU = parallel.gpu.CUDAKernel('KNNSearch.ptx','KNNSearch.cu');
    numPoints= size(voxelGridPts,1);
    numIterGPU = floor(numPoints/30000000) + 1;
    nnSqrDists = zeros(numPoints,1);
    for iterGPU = 1:numIterGPU
        if iterGPU ~= numIterGPU
            [nnIdx,tmpSqrDists] = multiQueryKNNSearchImplGPU(targetPointCloud,voxelGridPts(((iterGPU-1)*30000000+1):(iterGPU*30000000),:));
            nnSqrDists(((iterGPU-1)*30000000+1):(iterGPU*30000000)) = tmpSqrDists;
        else
            [nnIdx,tmpSqrDists] = multiQueryKNNSearchImplGPU(targetPointCloud,voxelGridPts(((iterGPU-1)*30000000+1):numPoints,:));
            nnSqrDists(((iterGPU-1)*30000000+1):numPoints) = tmpSqrDists;
        end
    end
    nnDists = sqrt(nnSqrDists)';

    voxelGridTDF = reshape(nnDists,size(voxelGridPtsX));

    % Truncate and flip voxel grid TDF values
    voxelGridTDF(find(voxelGridTDF > truncMargin)) = truncMargin;
    voxelGridTDF = voxelGridTDF./truncMargin;
    voxelGridTDF = 1 - voxelGridTDF;
    
    % Compute coordinates of voxel grid origin corner (0,0,0)
    voxelGridOrigin = min(voxelGridPts,[],1);
end





