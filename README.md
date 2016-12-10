# 3DMatch Toolbox
3DMatch is a ConvNet-based local geometric feature descriptor that operates on 3D data (i.e. point clouds, depth maps, meshes, etc.). This toolbox provides code to use 3DMatch for geometric registration and keypoint matching, as well as code to train 3DMatch from existing RGB-D reconstructions. This is the reference implementation of our paper:

### 3DMatch: Learning Local Geometric Descriptors from 3D Reconstructions ([pdf](), [arxiv](), [webpage](http://3dmatch.cs.princeton.edu/))

*Andy Zeng, Shuran Song, Matthias Nießner, Matthew Fisher, Jianxiong Xiao, and Thomas Funkhouser*

Matching local geometric features on real-world depth images is a challenging task due to the noisy, low-resolution, and incomplete nature of 3D scan data. These difficulties limit the performance of current state-of-art methods, which are typically based on histograms over geometric properties. In this paper, we present 3DMatch, a data-driven model that learns a local volumetric patch descriptor for establishing correspondences between partial 3D data. To amass training data for our model, we propose an unsupervised feature learning method that leverages the millions of correspondence labels found in existing RGB-D reconstructions. Experiments show that our descriptor is not only able to match local geometry in new scenes for reconstruction, but also generalize to different tasks and spatial scales (e.g. instance-level object model alignment for the Amazon Picking Challenge, and mesh surface correspondence). Results show that 3DMatch consistently outperforms other state-of-the-art approaches by a significant margin. 

![Teaser](teaser.png?raw=true)

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

## Table of Contents
* [Dependencies](#dependencies)
* [Quick Start Demo: Align Two Point Clouds with 3DMatch](#quick-start-demo-align-two-point-clouds-with-3dmatch)
* [Training 3DMatch from RGB-D Reconstructions](#training-3dmatch-from-rgb-d-reconstructions)
* [Multi-Frame Depth TSDF Fusion](#multi-frame-depth-tsdf-fusion)
* [Evaluation Code](#evaluation-code)

## Dependencies

Our reference implementation of 3DMatch, as well as other components in this toolbox, require the following dependencies. Tested on Ubuntu 14.04.

1. [CUDA 7.5](https://developer.nvidia.com/cuda-downloads) and [cuDNN 5](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA. Below are some additional steps to set up cuDNN 5. **NOTE** We highly recommend that you install different versions of cuDNN to different directories (e.g., ```/usr/local/cudnn/vXX```) because different software packages may require different versions.

	```shell
	LIB_DIR=lib$([[ $(uname) == "Linux" ]] && echo 64)
	CUDNN_LIB_DIR=/usr/local/cudnn/v5/$LIB_DIR
	echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_LIB_DIR >> ~/.profile && ~/.profile

	tar zxvf cudnn*.tgz
	sudo cp cuda/$LIB_DIR/* $CUDNN_LIB_DIR/
	sudo cp cuda/include/* /usr/local/cudnn/v5/include/
	```

2. OpenCV (tested with OpenCV 2.4.11)
 * Used for reading image files

3. Matlab 2015b or higher (tested with Matlab 2016a)

## Quick Start Demo: Align Two Point Clouds with 3DMatch

See folder `core`

A brief demo to show how to align two 3D point clouds (projected from single-view depth maps) using the 3DMatch descriptor (with Marvin) and standard RANSAC.

1. Navigate to folder `core/`
2. Run in terminal `./compile.sh` to compile `demo.cu` and Marvin
3. Run bash script `./download-weights.sh` to download our 3DMatch pre-trained weights
4. Run in terminal `./demo ../data/sample/3dmatch-demo/single-depth-1.ply fragment-1` to load the first example 3D point cloud and compute random surface keypoints and their 3DMatch descriptors (saved to binary files on disk)
5. Run in terminal `./demo ../data/sample/3dmatch-demo/single-depth-2.ply fragment-2` to do the same for the second example 3D point cloud
6. In Matlab, run `demo.m` to load the keypoints and 3DMatch descriptors and use RANSAC to register the two point clouds. The alignment result is saved as the file `result.ply` which can be viewed with Meshlab or any other 3D viewer.

Note: there is a small chance that alignment may fail on the first try of this demo due to bad keypoints, which are selected at random.

## Training 3DMatch from RGB-D Reconstructions

See folder `training`

Code for training 3DMatch with [Marvin](http://marvin.is/), a lightweight GPU-only neural network framework. Includes Siamese network architecture .json file `training/net.json` and a CUDA/C++ Marvin data layer in `training/match.hpp` that randomly samples correspondences from RGB-D reconstruction datasets (which can be downloaded from our [project webpage](http://3dmatch.cs.princeton.edu/)).

### Setup Instructions
Download one or more scenes from RGB-D reconstruction datasets on our [project webpage](http://3dmatch.cs.princeton.edu/). These datasets have been converted into a unified format, which is compatible with our Marvin data layer used to train 3DMatch. Save at least one scene into `data/train` and another scene into `data/test` such that the folder hierarchy looks something like this:
```shell
|——— training
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
2. Navigate to directory `training/`
3. Run in terminal `./compile.sh` to compile Marvin
4. Run in terminal `./marvin train net.json` to train a 3DMatch model from scratch over correspondences from the RGB-D scenes saved in `data/train`
5. To train 3DMatch using pre-trained weights from a Marvin tensor file, run in terminal `./marvin train net.json pre-trained-weights.marvin`

## Multi-Frame Depth TSDF Fusion

See folder `depth-fusion`

CUDA/C++ code to fuse multiple registered depth maps into a TSDF voxel volume ([Curless and Levoy 1996](http://graphics.stanford.edu/papers/volrange/volrange.pdf)), which can then be used to create surface meshes and point clouds.

### Demo

Fuses 50 registered depth maps from directory `data/sample/depth-fusion-demo/rgbd-frames` into a TSDF voxel volume, and creates a surface point cloud `tsdf.ply`

1. Navigate to directory `depth-fusion/`
2. Run in terminal `./compile.sh` to compile the demo code `demo.cu`
3. Run in terminal `./demo`

## Evaluation

See folder `evaluation`

Reference implementation for the experiments in our paper.

### Keypoint Matching Benchmark

### Geometric Registration Benchmark

See folder `evaluation/geometric-registration`

Includes Matlab code to run evaluation on the geometric registration benchmarks described [here](). Overview:
* `getKeyptsAndDesc.m` - generates intermediate data (TDF voxel volumes, keypoints, and 3DMatch descriptors) for the scene fragments. You can also download our pre-computed data [here](). 
* `runFragmentRegistration.m` - read intermediate data and run RANSAC-based registration for every pair of fragments. 
* `writeLog` - read registration results from every pair of fragments and create a .log file
* `evaluate.m` - compute precision and recall from .log files for evaluation

Quick start: run Matlab script `evaluation/geometric-registration/evaluate.m`

Note: the TDF voxel grids of the scene fragments from the synthetic benchmark were computed using the deprecated code for accurate TDF (see `deprecated/pointCloud2AccTDF.m`) 3DMatch weights fine-tuned on training fragments can be downloaded [here](http://vision.princeton.edu/projects/2016/3DMatch/downloads/weights/3dmatch-weights-snapshot-127000-fragments-6000.marvin).

### Model Fitting for 6D Object Pose Estimation in the Amazon Picking Challenge

See folder `evaluation/model-fitting-apc`

Includes code and pre-trained models to evaluate 3DMatch for model fitting on the Shelf & Tote dataset. For an evaluation example, run Matlab script `evaluation/model-fitting-apc/getError.m`

### Mesh Correspondence in Shape2Pose

See folder `evaluation/mesh-correspondence-shape2pose`

Includes code to generate mesh correspondence visualizations on the meshes from the [Shape2Pose dataset](http://gfx.cs.princeton.edu/gfx/pubs/Kim_2014_SHS/index.php). You can download our pre-computed data (TDF voxel grid volumes of the meshes, surface keypoints, 3DMatch descriptors) [here](http://vision.princeton.edu/projects/2016/3DMatch/downloads/shape2pose.zip). For a quick visualization, run the Matlab script `evaluation/mesh-correspondence-shape2pose/keypointRetrieval.m`.