# 3DMatch Toolbox
3DMatch is a ConvNet-based local geometric feature descriptor that operates on 3D data (i.e. point clouds, depth maps, meshes, etc.). This toolbox provides code to use 3DMatch for geometric registration and keypoint matching, as well as code to train 3DMatch from existing RGB-D reconstructions. This is the reference implementation of models and code for our paper:

### 3DMatch: Learning Local Geometric Descriptors from 3D Reconstructions ([pdf](), [arxiv](), [webpage](http://3dmatch.cs.princeton.edu/))

*Andy Zeng, Shuran Song, Matthias Nießner, Matthew Fisher, Jianxiong Xiao, and Thomas Funkhouser*

Matching local geometric features on real-world depth images is a challenging task due to the noisy, low-resolution, and incomplete nature of 3D scan data. These difficulties limit the performance of current state-of-art methods, which are typically based on histograms over geometric properties. In this paper, we present 3DMatch, a data-driven model that learns a local volumetric patch descriptor for establishing correspondences between partial 3D data. To amass training data for our model, we propose an unsupervised feature learning method that leverages the millions of correspondence labels found in existing RGB-D reconstructions. Experiments show that our descriptor is not only able to match local geometry in new scenes for reconstruction, but also generalize to different tasks and spatial scales (e.g. instance-level object model alignment for the Amazon Picking Challenge, and mesh surface correspondence). Results show that 3DMatch consistently outperforms other state-of-the-art approaches by a significant margin. 

![Teaser](teaser.png?raw=true)

#### Citing

If you find this code useful in your work, please consider citing:

```shell

```

#### License

This code is released under the Simplified BSD License (refer to the LICENSE file for details).

#### Datasets
All relevant dataset information and downloads can be found [here](http://3dmatch.cs.princeton.edu/).

#### Contact
If you have any questions or find any bugs, please let me know: [Andy Zeng](http://www.cs.princeton.edu/~andyz/) andyz[at]princeton[dot]edu

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
 * Used for reading images

3. Matlab 2015b or higher (tested with Matlab 2016a)

## Documentation
* [Training 3DMatch from RGB-D Reconstructions](#training-3dmatch-from-rgb-d-reconstructions)
* [Multi-Frame Depth TSDF Fusion](#multi-frame-depth-tsdf-fusion)





## 3DMatch for Geometric Registration





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

The demo fuses 50 registered depth maps from directory `data/sample/depth-fusion-demo/fragment-1-rgbd-frames` into a TSDF voxel volume, and creates a surface point cloud `tsdf.ply`

1. Navigate to `depth-fusion/`
2. Run in terminal `./compile.sh` to compile the demo code `demo.cu`
3. Run in terminal `./demo`

## Evaluation Code

See folder `evaluation`

Code and models used to perform the experiments in our paper.

### Geometric Registration (of Fused Scene Fragments)









