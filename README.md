# 3DMatch Toolbox
3DMatch is a ConvNet-based local geometric feature descriptor that operates on 3D data (i.e. point clouds, depth maps, meshes, etc.). This toolbox provides code to use 3DMatch for geometric registration and keypoint matching, as well as code to train 3DMatch from existing RGB-D reconstructions. This is the reference implementation of our paper:

### 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions

[PDF](https://arxiv.org/pdf/1603.08182.pdf) | [Webpage & Benchmarks & Datasets](http://3dmatch.cs.princeton.edu/) | [Video](https://www.youtube.com/watch?v=gZrsJJtDvvA)

*[Andy Zeng](http://andyzeng.com/), [Shuran Song](http://vision.princeton.edu/people/shurans/), [Matthias Nießner](http://www.niessnerlab.org/members/matthias_niessner/profile.html), [Matthew Fisher](https://research.adobe.com/person/matt-fisher/), [Jianxiong Xiao](http://vision.princeton.edu/people/xj/), and [Thomas Funkhouser](http://www.cs.princeton.edu/~funk/)*

IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017 **[Oral Presentation](https://www.youtube.com/watch?v=qNVZl7bCjsU&list=PL_bDvITUYucADb15njRd7geem8vxOyo6N&index=3)**

Matching local geometric features on real-world depth images is a challenging task due to the noisy, low-resolution, and incomplete nature of 3D scan data. These difficulties limit the performance of current state-of-art methods, which are typically based on histograms over geometric properties. In this paper, we present 3DMatch, a data-driven model that learns a local volumetric patch descriptor for establishing correspondences between partial 3D data. To amass training data for our model, we propose an unsupervised feature learning method that leverages the millions of correspondence labels found in existing RGB-D reconstructions. Experiments show that our descriptor is not only able to match local geometry in new scenes for reconstruction, but also generalize to different tasks and spatial scales (e.g. instance-level object model alignment for the Amazon Picking Challenge, and mesh surface correspondence). Results show that 3DMatch consistently outperforms other state-of-the-art approaches by a significant margin. 

![Overview](overview.jpg?raw=true)

#### Citing

If you find this code useful in your work, please consider citing:

```shell
@inproceedings{zeng20163dmatch, 
	title={3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions}, 
	author={Zeng, Andy and Song, Shuran and Nie{\ss}ner, Matthias and Fisher, Matthew and Xiao, Jianxiong and Funkhouser, Thomas}, 
	booktitle={CVPR}, 
	year={2017} 
}
```

#### License

This code is released under the Simplified BSD License (refer to the LICENSE file for details).

#### Benchmarks and Datasets
All relevant information and downloads can be found [here](http://3dmatch.cs.princeton.edu/).

#### Contact
If you have any questions or find any bugs, please let me know: [Andy Zeng](http://www.cs.princeton.edu/~andyz/) andyz[at]princeton[dot]edu

## Change Log
* **Mar. 20, 2018.** Update: added labels for test-set of keypoint matching benchmark (for convenience).
* **Nov. 02, 2017.** Bug fix: added `#include <random>` to utils.hpp in demo code.
* **Oct. 30, 2017.** Bug fix: included Quoc-Huy's fix for NaN errors that occasionally occur during training.
* **Oct. 28, 2017.** Notice: demo code only reads 3D point clouds saved in a simple binary format. If you would like to run the 3DMatch demo code on your own point cloud format, please modify demo.cu accordingly.
* **Apr. 06, 2017.** Notice: 3DMatch uses cuDNN 5.1. Revised install instructions.

## Dependencies

Our reference implementation of 3DMatch, as well as other components in this toolbox, require the following dependencies. Tested on Ubuntu 14.04.

0. [CUDA 7.5](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN 5.1](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA. Below are some additional steps to set up cuDNN 5.1. **NOTE** We highly recommend that you install different versions of cuDNN to different directories (e.g., ```/usr/local/cudnn/vXX```) because different software packages may require different versions.

	```shell
	LIB_DIR=lib$([[ $(uname) == "Linux" ]] && echo 64)
	CUDNN_LIB_DIR=/usr/local/cudnn/v5.1/$LIB_DIR
	echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_LIB_DIR >> ~/.profile && ~/.profile

	tar zxvf cudnn*.tgz
	sudo cp cuda/$LIB_DIR/* $CUDNN_LIB_DIR/
	sudo cp cuda/include/* /usr/local/cudnn/v5.1/include/
	```

0. OpenCV (tested with OpenCV 2.4.11)
 * Used for reading image files

0. Matlab 2015b or higher (tested with Matlab 2016a)

## Table of Contents
* [Demo: Align Two Point Clouds with 3DMatch](#demo-align-two-point-clouds-with-3dmatch)
* [Converting 3D Data to TDF Voxel Grids](#converting-3d-data-to-tdf-voxel-grids)
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

0. Load the two example 3D point clouds, compute their TDF voxel grid volumes, and compute random surface keypoints and their 3DMatch descriptors (saved to binary files on disk). Warning: this demo only reads 3D point clouds saved in a simple binary format. If you would like to run the 3DMatch demo code on your own point cloud format, please modify demo.cu accordingly.

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

## Converting 3D Data to TDF Voxel Grids

Instructions on how to convert from various 3D data representations into a voxel grid of Truncated Distance Function (TDF) values.

0. Point cloud to TDF voxel grid (using nearest neighbor point distances)
 * See [C++/CUDA demo code](https://github.com/andyzeng/3dmatch-toolbox/blob/master/core/demo.cu) (ComputeTDF) which approximates TDF values (fast) using an occupancy voxel grid.
 * Alternative: See [Matlab/CUDA code](https://github.com/andyzeng/3dmatch-toolbox/blob/master/deprecated/pointCloud2AccTDF.m) which computes accurate TDF values but is very slow.
 * Alternative: See [Matlab code](https://github.com/andyzeng/3dmatch-toolbox/blob/master/evaluation/model-fitting-apc/pointCloud2TDF.m) which also computes accurate TDF values, but works standalone on Matlab. Usually runs without memory problems if your point cloud is small.

0. Mesh to TDF voxel grid (using distance transform of mesh surface with [GAPS](https://github.com/tomfunkhouser/gaps)). Note that a version of GAPS is already included in this repository.
 * Instructions on installing GAPS and converting a sample mesh (.off file) into a voxel grid (binary .raw file of floats):

	```shell
	cd 3dmatch-toolbox/gaps

	# Install GAPS
	make

	# Run msh2df on example mesh file (see comments in msh2df.cpp for more instructions)
	cd bin/x86_64
	wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/gaps/bicycle000002.off
	./msh2df bicycle000002.off bicycle000002.raw -v # see comments in msh2df.cpp for more arguments

	# Download visualization script
	wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/gaps/showTDF.m
	```
 * Run the visualization script in Matlab

	```matlab
	% Visualize TDF voxel grid of mesh
	showTDF;
	```

0. Depth map to TDF voxel grid
 * Project depth map into a point cloud in 3D camera space and convert from point cloud to TDF voxel grid (see above)
 * Alternative: Convert from depth map(s) into a TSDF volume (see instructions [here](#multi-frame-depth-tsdf-fusion)) and compute the absolute value of each voxel (aka. projective TDF values, which behave differently near the view boundaries and regions of missing depth)

## Training 3DMatch from RGB-D Reconstructions

See folder `3dmatch-toolbox/training`

Code for training 3DMatch with [Marvin](http://marvin.is/), a lightweight GPU-only neural network framework. Includes Siamese network architecture .json file `training/net.json` and a CUDA/C++ Marvin data layer in `training/match.hpp` that randomly samples correspondences from RGB-D reconstruction datasets (which can be downloaded from our [project webpage](http://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets)).

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

### Additional Setup Instructions
You can download more scenes from RGB-D reconstruction datasets on our [project webpage](http://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets). These datasets have been converted into a unified format, which is compatible with our Marvin data layer used to train 3DMatch. Save at least one scene into `data/train` and another scene into `data/test` such that the folder hierarchy looks something like this:

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

## Multi-Frame Depth TSDF Fusion

See folder `3dmatch-toolbox/depth-fusion`

CUDA/C++ code to fuse multiple registered depth maps into a TSDF voxel volume ([Curless and Levoy 1996](http://graphics.stanford.edu/papers/volrange/volrange.pdf)), which can then be used to create surface meshes and point clouds.

### Demo

This demo fuses 50 registered depth maps from directory `data/sample/depth-fusion-demo/rgbd-frames` into a TSDF voxel volume, and creates a surface point cloud `tsdf.ply`

```shell
cd 3dmatch-toolbox/depth-fusion
./compile.sh
./demo # output saved to tsdf.ply
```

## Evaluation Code

See folder `3dmatch-toolbox/evaluation`

Evaluation code for the [Keypoint Matching Benchmark](http://3dmatch.cs.princeton.edu/#keypoint-matching-benchmark) and [Geometric Registration Benchmark](http://3dmatch.cs.princeton.edu/#geometric-registration-benchmark), as well as a reference implementation for the experiments in our [paper](https://arxiv.org/pdf/1603.08182.pdf).

### Keypoint Matching Benchmark

See folder `3dmatch-toolbox/evaluation/keypoint-matching`

Benchmark description and leaderboard can be found [here](http://3dmatch.cs.princeton.edu/#keypoint-matching-benchmark).

#### Evaluation Example

0. Navigate to `3dmatch-toolbox/evaluation/keypoint-matching` and run the following in Matlab:

	```matlab
	% Evaluate 3DMatch (3dmatch.log) on the validation set (validation-set-gt.log)
	getError;
	```

#### Run 3DMatch on the validation set to generate a .log file (3dmatch.log)


0. Compile C++/CUDA code to compute 3DMatch descriptors with Marvin

	```shell
	cd 3dmatch-toolbox/evaluation/keypoint-matching
	./compile.sh
	```

0. Download our 3DMatch pre-trained weights 

	```shell
	./download-weights.sh # 3dmatch-weights-snapshot-137000.marvin
	```

0. Download the validation set and test set

	```shell
	./download-validation.sh # validation-set.mat
	./download-test.sh # test-set.mat
	```

0. Modify and run the following script in Matlab:

	```matlab
	% Runs 3DMatch on the validation set and generates 3dmatch.log
	test3DMatch;
	```

#### Generate your own correspondence dataset from [RGB-D reconstructions](http://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets)

0. Download one or more scenes from RGB-D reconstruction datasets on our [project webpage](http://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets). Organize the [folder hierarchy as above](#additional-setup-instructions).

0. Modify and run the following script in Matlab:

	```matlab
	makeCorresDataset;
	```

### Geometric Registration Benchmark

See folder `3dmatch-toolbox/evaluation/geometric-registration`

Includes Matlab code to run evaluation on the geometric registration benchmarks described [here](http://3dmatch.cs.princeton.edu/#geometric-registration-benchmark). Overview:
* `getKeyptsAndDesc.m` - generates intermediate data (TDF voxel volumes, keypoints, and 3DMatch descriptors) for the scene fragments. You can also download our pre-computed data [here](http://3dmatch.cs.princeton.edu/#geometric-registration-synthetic-data). 
* `runFragmentRegistration.m` - read intermediate data and run RANSAC-based registration for every pair of fragments. 
* `writeLog` - read registration results from every pair of fragments and create a .log file
* `evaluate.m` - compute precision and recall from .log files for evaluation

#### Evaluation Example

Run the following in Matlab:

```matlab
% Evaluate 3DMatch on the geometric registration benchmark
evaluate;
```

Note: the TDF voxel grids of the scene fragments from the synthetic benchmark were computed using the deprecated code for accurate TDF (see `deprecated/pointCloud2AccTDF.m`). 3DMatch pre-trained weights fine-tuned on training fragments can be downloaded [here](http://vision.princeton.edu/projects/2016/3DMatch/downloads/weights/3dmatch-weights-snapshot-127000-fragments-6000.marvin).

### Model Fitting for 6D Object Pose Estimation in the Amazon Picking Challenge

See folder `3dmatch-toolbox/evaluation/model-fitting-apc`

Includes code and pre-trained models to evaluate 3DMatch for model fitting on the [Shelf & Tote dataset](http://www.cs.princeton.edu/~andyz/apc2016). You can download our pre-computed data (TDF voxel grid volumes for objects and scans, surface keypoints, descriptors, and pose predictions) [here](http://vision.princeton.edu/projects/2016/3DMatch/downloads/apc-intermediate-data.zip). For an evaluation example, run Matlab script `getError.m`

### Mesh Correspondence in Shape2Pose

See folder `3dmatch-toolbox/evaluation/mesh-correspondence-shape2pose`

Includes code to generate mesh correspondence visualizations on the meshes from the [Shape2Pose dataset](http://gfx.cs.princeton.edu/gfx/pubs/Kim_2014_SHS/index.php) using 3DMatch. You can also download our pre-computed data (TDF voxel grid volumes of the meshes, surface keypoints, 3DMatch descriptors) [here](http://vision.princeton.edu/projects/2016/3DMatch/downloads/shape2pose.zip). For a quick visualization, run the Matlab script `keypointRetrieval.m`.