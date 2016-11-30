# 3DMatch Toolbox
3DMatch is a ConvNet-based local geometric feature descriptor that operates on 3D data (i.e. point clouds, depth maps, meshes, etc.).









## Training 3DMatch from RGB-D Reconstructions

See folder `training/`

Code for training 3DMatch with [Marvin](http://marvin.is/), a lightweight GPU-only neural network framework. Includes Siamese network architecture .json file `training/net.json` and a CUDA/C++ Marvin data layer in `training/match.hpp` that randomly samples correspondences from RGB-D reconstruction datasets (which can be downloaded from our [project webpage](http://3dmatch.cs.princeton.edu/)).

### Dependencies

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

### Setup Instructions
1. Download one or more scenes from RGB-D reconstruction datasets on our [project webpage](http://3dmatch.cs.princeton.edu/). These datasets have been converted into a unified format, which is compatible with our Marvin data layer used to train 3DMatch. Save at least one scene into 'data/train' and another scene into `data/test` such that the folder hierarchy looks something like this:
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
3. Run in terminal `./compile.sh` to compile Marvin.
4. Run in terminal `./marvin train net.json` to train a 3DMatch model from scratch over correspondences from the RGB-D scenes saved in `data/train`.
5. To train 3DMatch using pre-trained weights from a Marvin tensor file, run in terminal `./marvin train net.json pre-trained-weights.marvin`.














