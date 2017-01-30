#define DATATYPE 0
#include "marvin.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char * argv[]) {

	std::cout << argv[1] << std::endl;
	std::string filename(argv[1]);

	// Read binary file containing TDF voxel grid values
    FILE * fp = fopen(argv[1],"rb");
    float numVolumesf;
    float volumeGridSizeXf;
    float volumeGridSizeYf;
    float volumeGridSizeZf;
	int iret = fread((void*)(&numVolumesf), sizeof(float), 1, fp);
	int numVolumes = (int)numVolumesf;
	iret = fread((void*)(&volumeGridSizeXf), sizeof(float), 1, fp);
	iret = fread((void*)(&volumeGridSizeYf), sizeof(float), 1, fp);
	iret = fread((void*)(&volumeGridSizeZf), sizeof(float), 1, fp);
    int numVolumeGridPoints = (int)(volumeGridSizeXf * volumeGridSizeYf * volumeGridSizeZf);
    float * tudfVoxf = new float[numVolumes*numVolumeGridPoints];
	iret = fread((void*)(tudfVoxf), sizeof(float), numVolumes*numVolumeGridPoints, fp);
	fclose(fp);

    // Start Marvin network
	marvin::Net convnet("tmp.json");
	convnet.Malloc(marvin::Testing);
	convnet.loadWeights("3dmatch-weights-snapshot-127000-apc-16000.marvin");
	marvin::Response * rData;
	marvin::Response * rFeat;
	rData = convnet.getResponse("data");
	rFeat = convnet.getResponse("feat");

	fp = fopen("feat.bin", "wb");
	fwrite(&numVolumesf, sizeof(float), 1, fp);
	for (int i = 1; i < 5; i++) {
		float value = (float)(rFeat->dim[i]);
		fwrite(&value, sizeof(float), 1, fp);
	}

	std::cout << numVolumes << std::endl;
    StorageT * itudfVox = new StorageT[50*numVolumeGridPoints];
	int numFeatVolumeGridPoints = rFeat->dim[0] * rFeat->dim[1] * rFeat->dim[2] * rFeat->dim[3] * rFeat->dim[4];
	StorageT * featVolume = new StorageT[numFeatVolumeGridPoints];

	for (int vox_idx = 0; vox_idx < numVolumes/50; ++vox_idx) {
	    for (int i = 0; i < 50*numVolumeGridPoints; ++i) {
	    	itudfVox[i] = CPUCompute2StorageT(tudfVoxf[vox_idx*50*numVolumeGridPoints + i]);
		}
	    
		cudaMemcpy(rData->dataGPU, itudfVox, rData->numBytes(), cudaMemcpyHostToDevice);
		marvin::checkCUDA(__LINE__, cudaGetLastError());

		convnet.forward();

		cudaMemcpy(featVolume, rFeat->dataGPU, rFeat->numBytes(), cudaMemcpyDeviceToHost);

		for (int i = 0; i < numFeatVolumeGridPoints; i++) {
			float value = CPUStorage2ComputeT(featVolume[i]);
			fwrite(&value, sizeof(float), 1, fp);
		}

	}
	fclose(fp);

	return 0;
}
















































