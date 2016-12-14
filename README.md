# 3DMatch Toolbox
3DMatch is a ConvNet-based local geometric feature descriptor that operates on 3D data (i.e. point clouds, depth maps, meshes, etc.). This toolbox provides code to use 3DMatch for geometric registration and keypoint matching, as well as code to train 3DMatch from existing RGB-D reconstructions. This is the reference implementation of our paper:

### 3DMatch: Learning Local Geometric Descriptors from 3D Reconstructions ([pdf](), [webpage](http://3dmatch.cs.princeton.edu/))

*Andy Zeng, Shuran Song, Matthias Nießner, Matthew Fisher, Jianxiong Xiao, and Thomas Funkhouser*

Matching local geometric features on real-world depth images is a challenging task due to the noisy, low-resolution, and incomplete nature of 3D scan data. These difficulties limit the performance of current state-of-art methods, which are typically based on histograms over geometric properties. In this paper, we present 3DMatch, a data-driven model that learns a local volumetric patch descriptor for establishing correspondences between partial 3D data. To amass training data for our model, we propose an unsupervised feature learning method that leverages the millions of correspondence labels found in existing RGB-D reconstructions. Experiments show that our descriptor is not only able to match local geometry in new scenes for reconstruction, but also generalize to different tasks and spatial scales (e.g. instance-level object model alignment for the Amazon Picking Challenge, and mesh surface correspondence). Results show that 3DMatch consistently outperforms other state-of-the-art approaches by a significant margin. 

![Overview](overview.jpg?raw=true)

#### Citing

If you find this code useful in your work, please consider citing:

```shell
@article{zeng20163dmatch, 
	title={3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions}, 
	author={Zeng, Andy and Song, Shuran and Nie{\ss}ner, Matthias and Fisher, Matthew and Xiao, Jianxiong and Funkhouser, Thomas}, 
	journal={arXiv preprint arXiv:1603.08182}, 
	year={2016} 
}
```

#### License

This code is released under the Simplified BSD License (refer to the LICENSE file for details).

#### Benchmarks and Datasets
All relevant information and downloads can be found [here](http://3dmatch.cs.princeton.edu/).

#### Contact
If you have any questions or find any bugs, please let me know: [Andy Zeng](http://www.cs.princeton.edu/~andyz/) andyz[at]princeton[dot]edu

## Dependencies

Our reference implementation of 3DMatch, as well as other components in this toolbox, require the following dependencies. Tested on Ubuntu 14.04.

0. [CUDA 7.5](https://developer.nvidia.com/cuda-downloads) and [cuDNN 5](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA. Below are some additional steps to set up cuDNN 5. **NOTE** We highly recommend that you install different versions of cuDNN to different directories (e.g., ```/usr/local/cudnn/vXX```) because different software packages may require different versions.

	```shell
	LIB_DIR=lib$([[ $(uname) == "Linux" ]] && echo 64)
	CUDNN_LIB_DIR=/usr/local/cudnn/v5/$LIB_DIR
	echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_LIB_DIR >> ~/.profile && ~/.profile

	tar zxvf cudnn*.tgz
	sudo cp cuda/$LIB_DIR/* $CUDNN_LIB_DIR/
	sudo cp cuda/include/* /usr/local/cudnn/v5/include/
	```

0. OpenCV (tested with OpenCV 2.4.11)
 * Used for reading image files

0. Matlab 2015b or higher (tested with Matlab 2016a)

## Table of Contents
* [Demo: Align Two Point Clouds with 3DMatch](#demo-align-two-point-clouds-with-3dmatch)
* [Training 3DMatch from RGB-D Reconstructions](#training-3dmatch-from-rgb-d-reconstructions)
* [Multi-Frame Depth TSDF Fusion](#multi-frame-depth-tsdf-fusion)
* [Evaluation Code](#evaluation-code)

## Demo: Align Two Point Clouds with 3DMatch

![Demo-Teaser](demo-teaser.jpg?raw=true)

This demo aligns two 3D point clouds (projected from single-view depth maps) using our pre-trained 3DMatch descriptor (with Marvin) and standard RANSAC.

### Instructions

0. Checkout 3DMatch toolbox, compile C++/CUDA demo code and Marvin

	```shell
	git clone https://github.com/andyzeng/3dmatch-toolbox.git 3dmatch-toolbox
	cd 3dmatch-toolbox/core
	./compile.sh
	```

0. Download our 3DMatch pre-trained weights

	```shell
	./download-weights.sh # 3dmatch-weights-snapshot-137000.marvin
	```

0. Load the two example 3D point clouds and compute random surface keypoints and their 3DMatch descriptors (saved to binary files on disk)

	```shell
	# Generate fragment-1.desc.3dmatch.bin and fragment-1.keypts.bin
	./demo ../data/sample/3dmatch-demo/single-depth-1.ply fragment-1

	# Generate fragment-2.desc.3dmatch.bin and fragment-2.keypts.bin
	./demo ../data/sample/3dmatch-demo/single-depth-2.ply fragment-2 
	```

0. Run the following script in Matlab:

	```matlab
	% Load keypoints and 3DMatch descriptors and use RANSAC to register the two
	% point clouds. A visualization of the aligned point clouds is saved into
	% the file `result.ply` which can be viewed with Meshlab or any other 3D
	% viewer. Note: there is a chance that alignment may fail on the first try
	% of this demo due to bad keypoints, which are selected randomly by default.
	demo;
	```

## Training 3DMatch from RGB-D Reconstructions

See folder `3dmatch-toolbox/training`

Code for training 3DMatch with [Marvin](http://marvin.is/), a lightweight GPU-only neural network framework. Includes Siamese network architecture .json file `training/net.json` and a CUDA/C++ Marvin data layer in `training/match.hpp` that randomly samples correspondences from RGB-D reconstruction datasets (which can be downloaded from our [project webpage](http://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets)).

### Additional Setup Instructions
0. Download one or more scenes from RGB-D reconstruction datasets on our [project webpage](http://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets). These datasets have been converted into a unified format, which is compatible with our Marvin data layer used to train 3DMatch. Save at least one scene into `data/train` and another scene into `data/test` such that the folder hierarchy looks something like this:

	```shell
	|——— training
	     |——— core
	          |——— marvin.hpp
	          |——— ...
	|——— data
	     |——— train
	          |——— rgbd-dataset-scene-1
	               |——— seq-01
	               |——— seq-02
	               |——— camera-intrinsics.txt
	               |——— ...
	          |——— ...
	     |——— test
	          |——— rgbd-dataset-scene-2
	               |——— seq-01
	               |——— camera-intrinsics.txt
	               |——— ...
	```

### Quick Start
0. Compile Marvin
	
	```shell
	cd 3dmatch-toolbox/training
	./compile.sh
	```

0. Download several training and testing scenes from RGB-D reconstruction datasets (download more scenes [here](http://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets))

	```shell
	cd ../data
	mkdir train && mkdir test && mkdir backup
	cd train
	wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-brown_cogsci_1-brown_cogsci_1.zip
	wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/7-scenes-heads.zip
	wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-harvard_c11-hv_c11_2.zip
	unzip sun3d-brown_cogsci_1-brown_cogsci_1.zip
	unzip 7-scenes-heads.zip
	unzip sun3d-harvard_c11-hv_c11_2.zip
	mv *.zip ../backup
	cd ../test
	wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-hotel_umd-maryland_hotel3.zip
	unzip sun3d-hotel_umd-maryland_hotel3.zip
	mv *.zip ../backup
	cd ../../training
	```

0. Train a 3DMatch model from scratch over correspondences from the RGB-D scenes saved in `data/train`

	```shell
	./marvin train net.json
	```

0. (Optional) Train 3DMatch using pre-trained weights from a Marvin tensor file

	```shell
	./marvin train net.json your-pre-trained-weights.marvin
	```

## Multi-Frame Depth TSDF Fusion

See folder `3dmatch-toolbox/depth-fusion`

CUDA/C++ code to fuse multiple registered depth maps into a TSDF voxel volume ([Curless and Levoy 1996](http://graphics.stanford.edu/papers/volrange/volrange.pdf)), which can then be used to create surface meshes and point clouds.

### Demo

Fuses 50 registered depth maps from directory `data/sample/depth-fusion-demo/rgbd-frames` into a TSDF voxel volume, and creates a surface point cloud `tsdf.ply`

```shell
cd 3dmatch-toolbox/depth-fusion
./compile.sh
./demo # output saved to tsdf.ply
```

## Evaluation Code

See folder `3dmatch-toolbox/evaluation`

Evaluation code for our [Keypoint Matching Benchmark](http://3dmatch.cs.princeton.edu/#keypoint-matching-benchmark) and [Geometric Registration Benchmark](http://3dmatch.cs.princeton.edu/#geometric-registration-benchmark), as well as the reference implementation for the experiments in our [paper](https://arxiv.org/abs/1603.08182).

### Keypoint Matching Benchmark

See folder `3dmatch-toolbox/evaluation/keypoint-matching`

Includes Matlab code to generate a correspondence dataset from the RGB-D reconstructions [here](), as well as code to run evaluation on the keypoint matching benchmarks described [here](). Overview:

#### Quick Start

Run the following in Matlab:

```matlab
% Evaluate 3DMatch on the keypoint matching benchmark
evaluate;
```

### Geometric Registration Benchmark

See folder `3dmatch-toolbox/evaluation/geometric-registration`

Includes Matlab code to run evaluation on the geometric registration benchmarks described [here](http://3dmatch.cs.princeton.edu/#geometric-registration-benchmark). Overview:
* `getKeyptsAndDesc.m` - generates intermediate data (TDF voxel volumes, keypoints, and 3DMatch descriptors) for the scene fragments. You can also download our pre-computed data [here](). 
* `runFragmentRegistration.m` - read intermediate data and run RANSAC-based registration for every pair of fragments. 
* `writeLog` - read registration results from every pair of fragments and create a .log file
* `evaluate.m` - compute precision and recall from .log files for evaluation

#### Quick Start

Run the following in Matlab:

```matlab
% Evaluate 3DMatch on the geometric registration benchmark
evaluate;
```

Note: the TDF voxel grids of the scene fragments from the synthetic benchmark were computed using the deprecated code for accurate TDF (see `deprecated/pointCloud2AccTDF.m`). 3DMatch pre-trained weights fine-tuned on training fragments can be downloaded [here](http://vision.princeton.edu/projects/2016/3DMatch/downloads/weights/3dmatch-weights-snapshot-127000-fragments-6000.marvin).

### Model Fitting for 6D Object Pose Estimation in the Amazon Picking Challenge

See folder `3dmatch-toolbox/evaluation/model-fitting-apc`

Includes code and pre-trained models to evaluate 3DMatch for model fitting on the Shelf & Tote dataset. For an evaluation example, run Matlab script `getError.m`

### Mesh Correspondence in Shape2Pose

See folder `3dmatch-toolbox/evaluation/mesh-correspondence-shape2pose`

Includes code to generate mesh correspondence visualizations on the meshes from the [Shape2Pose dataset](http://gfx.cs.princeton.edu/gfx/pubs/Kim_2014_SHS/index.php) using 3DMatch. You can also download our pre-computed data (TDF voxel grid volumes of the meshes, surface keypoints, 3DMatch descriptors) [here](http://vision.princeton.edu/projects/2016/3DMatch/downloads/shape2pose.zip). For a quick visualization, run the Matlab script `keypointRetrieval.m`.