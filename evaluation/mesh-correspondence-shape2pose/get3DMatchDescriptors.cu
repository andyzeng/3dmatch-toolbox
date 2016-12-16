#define DATATYPE 0
#include "marvin.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char * argv[]) {

	std::string TDFFile(argv[1]);
	std::string keypointsFile(argv[2]);

	// Read binary file containing TDF voxel grid values
	FILE * fp = fopen(TDFFile.c_str(), "rb");
	float voxelGridSize[3];
	int iret = fread((void*)(voxelGridSize), sizeof(float), 3, fp);
	std::cout << voxelGridSize[0] << " " << voxelGridSize[1] << " " << voxelGridSize[2] << std::endl;
	float * TDFVoxelGrid = new float[(int)(voxelGridSize[0]) * (int)(voxelGridSize[1]) * (int)(voxelGridSize[2])];
	iret = fread((void*)(TDFVoxelGrid), sizeof(float), (int)(voxelGridSize[0]) * (int)(voxelGridSize[1]) * (int)(voxelGridSize[2]), fp);
	fclose(fp);

	// Read binary file containing surface keypoints
	float numKeypoints;
	fp = fopen(keypointsFile.c_str(), "rb");
	iret = fread((void*)(&numKeypoints), sizeof(float), 1, fp);
	std::cout << numKeypoints << std::endl;
	float * keypoints = new float[3 * (int)(numKeypoints)];
	iret = fread((void*)(keypoints), sizeof(float), 3 * (int)(numKeypoints), fp);
	fclose(fp);

	// Start Marvin network
	marvin::Net convnet("tmp.json");
	convnet.Malloc(marvin::Testing);
	convnet.loadWeights("3dmatch-weights-snapshot-131000.marvin");
	marvin::Response * rData;
	marvin::Response * rFeat;
	rData = convnet.getResponse("data");
	rFeat = convnet.getResponse("feat");

	std::string resultsFilename = keypointsFile.substr(0, keypointsFile.length() - 4) + ".3dmatch.descriptors.bin";
	fp = fopen(resultsFilename.c_str(), "wb");
	float descriptorSize = 512.0f;
	fwrite(&numKeypoints, sizeof(float), 1, fp);
	fwrite(&descriptorSize, sizeof(float), 1, fp);

	// Load local 3D patches into batches
	int batchSize = 50;
	StorageT * batchTDF = new StorageT[batchSize * 30 * 30 * 30];
	for (int batchIdx = 0; batchIdx < numKeypoints / batchSize; batchIdx++) {
		for (int keypointIdx = batchIdx * batchSize; keypointIdx < (batchIdx + 1) * batchSize; keypointIdx++) {
			int localVoxelGridIdx = keypointIdx - batchIdx * batchSize;
			float keypointX = round(keypoints[3 * keypointIdx + 0]);
			float keypointY = round(keypoints[3 * keypointIdx + 1]);
			float keypointZ = round(keypoints[3 * keypointIdx + 2]);
			std::cout << keypointIdx << " " << localVoxelGridIdx << ": " << keypointX << " " << keypointY << " " << keypointZ << std::endl;
			StorageT * localTDF = new StorageT[30 * 30 * 30];
			int localVoxelIdx = 0;
			for (int z = keypointZ - 15; z < keypointZ + 15; ++z)
				for (int y = keypointY - 15; y < keypointY + 15; ++y)
					for (int x = keypointX - 15; x < keypointX + 15; ++x) {
						localTDF[localVoxelIdx] = CPUCompute2StorageT(TDFVoxelGrid[z * (int)voxelGridSize[1] * (int)voxelGridSize[0] + y * (int)voxelGridSize[0] + x]);
						localVoxelIdx++;
					}
			for (int voxelIdx = 0; voxelIdx < 30 * 30 * 30; ++voxelIdx)
				batchTDF[localVoxelGridIdx * 30 * 30 * 30 + voxelIdx] = localTDF[voxelIdx];
			delete [] localTDF;
		}

		cudaMemcpy(rData->dataGPU, batchTDF, rData->numBytes(), cudaMemcpyHostToDevice);
		marvin::checkCUDA(__LINE__, cudaGetLastError());

		// Marvin forward pass through 3DMatch
		convnet.forward();

		// Save 3DMatch descriptors
		StorageT * descriptorValues = new StorageT[batchSize * (int)descriptorSize];
		cudaMemcpy(descriptorValues, rFeat->dataGPU, rFeat->numBytes(), cudaMemcpyDeviceToHost);
		for (int localVoxelGridIdx = 0; localVoxelGridIdx < batchSize; ++localVoxelGridIdx) {
			for (int descriptorValueIdx = 0; descriptorValueIdx < (int)descriptorSize; descriptorValueIdx++) {
				float value = CPUStorage2ComputeT(descriptorValues[localVoxelGridIdx * (int)descriptorSize + descriptorValueIdx]);
				fwrite(&value, sizeof(float), 1, fp);
			}
		}
		delete [] descriptorValues;
	}
	fclose(fp);

	return 0;
}