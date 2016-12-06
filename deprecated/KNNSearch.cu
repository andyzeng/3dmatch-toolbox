__global__ void KNNSearch( float * result, const int * args, const float * pc1, const float * pc2)
{
    
    int cudaNumBlocks = args[0];
    int cudaNumThreads = args[1];
    int pc1NumPts = args[2];
    int pc2NumPts = args[3];

    int pc2Idx = blockIdx.x * cudaNumThreads + threadIdx.x;

    float currPtX = pc2[pc2NumPts * 0 + pc2Idx];
    float currPtY = pc2[pc2NumPts * 1 + pc2Idx];
    float currPtZ = pc2[pc2NumPts * 2 + pc2Idx];

    if (pc2Idx < pc2NumPts) {
      int nnIdx = 0;
      float nnDist = 100000.0f;
      for (int i = 0; i < pc1NumPts; i++) {
        float otherPtX = pc1[pc1NumPts * 0 + i];
        float otherPtY = pc1[pc1NumPts * 1 + i];
        float otherPtZ = pc1[pc1NumPts * 2 + i];

        float checkDist = (currPtX - otherPtX) * (currPtX - otherPtX) +
                                (currPtY - otherPtY) * (currPtY - otherPtY) +
                                (currPtZ - otherPtZ) * (currPtZ - otherPtZ);

        if (checkDist < nnDist) {
          nnDist = checkDist;
          nnIdx = i;
        }

      }




      result[pc2NumPts * 0 + pc2Idx] = nnIdx + 1;
      result[pc2NumPts * 1 + pc2Idx] = nnDist;


    }
}