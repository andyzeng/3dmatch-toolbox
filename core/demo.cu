// ---------------------------------------------------------
// Copyright (c) 2016, Andy Zeng
//
// This file is part of the 3DMatch Toolbox and is available
// under the terms of the Simplified BSD License provided in
// LICENSE. Please retain this notice and LICENSE if you use
// this file (or any portion of it) in your project.
// ---------------------------------------------------------

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "marvin.hpp"

#define CUDA_NUM_THREADS 512
#define CUDA_MAX_NUM_BLOCKS 2880

// CUDA kernel function to compute TDF voxel grid values given a point cloud (warning: approximate, but fast)
__global__
void ComputeTDF(int CUDA_LOOP_IDX, float * voxel_grid_occ, float * voxel_grid_TDF,
                int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                float voxel_size, float trunc_margin) {

  int voxel_idx = CUDA_LOOP_IDX * CUDA_NUM_THREADS * CUDA_MAX_NUM_BLOCKS + blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
  if (voxel_idx > (voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z))
    return;

  int pt_grid_z = (int)floor((float)voxel_idx / ((float)voxel_grid_dim_x * (float)voxel_grid_dim_y));
  int pt_grid_y = (int)floor(((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y)) / (float)voxel_grid_dim_x);
  int pt_grid_x = (int)((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y) - ((float)pt_grid_y * (float)voxel_grid_dim_x));

  int search_radius = (int)round(trunc_margin / voxel_size);

  if (voxel_grid_occ[voxel_idx] > 0) {
    voxel_grid_TDF[voxel_idx] = 1.0f; // on surface
    return;
  }

  // Find closest surface point
  for (int iix = max(0, pt_grid_x - search_radius); iix < min(voxel_grid_dim_x, pt_grid_x + search_radius + 1); ++iix)
    for (int iiy = max(0, pt_grid_y - search_radius); iiy < min(voxel_grid_dim_y, pt_grid_y + search_radius + 1); ++iiy)
      for (int iiz = max(0, pt_grid_z - search_radius); iiz < min(voxel_grid_dim_z, pt_grid_z + search_radius + 1); ++iiz) {
        int iidx = iiz * voxel_grid_dim_x * voxel_grid_dim_y + iiy * voxel_grid_dim_x + iix;
        if (voxel_grid_occ[iidx] > 0) {
          float xd = (float)(pt_grid_x - iix);
          float yd = (float)(pt_grid_y - iiy);
          float zd = (float)(pt_grid_z - iiz);
          float dist = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_radius;
          if ((1.0f - dist) > voxel_grid_TDF[voxel_idx])
            voxel_grid_TDF[voxel_idx] = 1.0f - dist;
        }
      }
}

// Demo code to show how to generate keypoints and 3DMatch descriptors from a point cloud
// 1. Loads a point cloud file
// 2. Generates a TDF voxel volume for the point cloud
// 3. Finds random surface keypoints
// 4. Compute 3DMatch descriptor vectors for all keypoints using Marvin
int main(int argc, char * argv[]) {

  std::string pointcloud_filename(argv[1]);
  std::string out_prefix_filename(argv[2]);

  // Super hacky code to read a point cloud file (replace this...)
  std::ifstream pointcloud_file(pointcloud_filename.c_str());
  if (!pointcloud_file) {
    std::cerr << "Point cloud file not found." << std::endl;
    return -1;
  }
  int num_pts = 0;
  for (int line_idx = 0; line_idx < 7; ++line_idx) {
    std::string line_str;
    std::getline(pointcloud_file, line_str);
    if (line_idx == 2) {
      std::istringstream tmp_line(line_str);
      std::string tmp_line_prefix;
      tmp_line >> tmp_line_prefix;
      tmp_line >> tmp_line_prefix;
      tmp_line >> num_pts;
    }
  }
  if (num_pts == 0)
    return 0;
  float * pts = new float[num_pts * 3]; // Nx3 matrix saved as float array (row-major order)
  pointcloud_file.read((char*)pts, sizeof(float) * num_pts * 3);
  pointcloud_file.close();

  std::cout << "Loaded point cloud with " << num_pts << " points!" << std::endl;

  float voxel_size = 0.01;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_padding = 15; // in voxels

  // Compute point cloud coordinates of the origin voxel (0,0,0) of the voxel grid
  float voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z;
  float voxel_grid_max_x, voxel_grid_max_y, voxel_grid_max_z;
  voxel_grid_origin_x = pts[0]; voxel_grid_max_x = pts[0];
  voxel_grid_origin_y = pts[1]; voxel_grid_max_y = pts[1];
  voxel_grid_origin_z = pts[2]; voxel_grid_max_z = pts[2];
  for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
    voxel_grid_origin_x = min(voxel_grid_origin_x, pts[pt_idx * 3 + 0]);
    voxel_grid_origin_y = min(voxel_grid_origin_y, pts[pt_idx * 3 + 1]);
    voxel_grid_origin_z = min(voxel_grid_origin_z, pts[pt_idx * 3 + 2]);
    voxel_grid_max_x = max(voxel_grid_max_x, pts[pt_idx * 3 + 0]);
    voxel_grid_max_y = max(voxel_grid_max_y, pts[pt_idx * 3 + 1]);
    voxel_grid_max_z = max(voxel_grid_max_z, pts[pt_idx * 3 + 2]);
  }

  int voxel_grid_dim_x = round((voxel_grid_max_x - voxel_grid_origin_x) / voxel_size) + 1 + voxel_grid_padding * 2;
  int voxel_grid_dim_y = round((voxel_grid_max_y - voxel_grid_origin_y) / voxel_size) + 1 + voxel_grid_padding * 2;
  int voxel_grid_dim_z = round((voxel_grid_max_z - voxel_grid_origin_z) / voxel_size) + 1 + voxel_grid_padding * 2;

  voxel_grid_origin_x = voxel_grid_origin_x - voxel_grid_padding * voxel_size + voxel_size / 2;
  voxel_grid_origin_y = voxel_grid_origin_y - voxel_grid_padding * voxel_size + voxel_size / 2;
  voxel_grid_origin_z = voxel_grid_origin_z - voxel_grid_padding * voxel_size + voxel_size / 2;

  std::cout << "Size of TDF voxel grid: " << voxel_grid_dim_x << " x " << voxel_grid_dim_y << " x " << voxel_grid_dim_z << std::endl;
  std::cout << "Computing TDF voxel grid..." << std::endl;

  // Compute surface occupancy grid
  float * voxel_grid_occ = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  memset(voxel_grid_occ, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
  for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
    int pt_grid_x = round((pts[pt_idx * 3 + 0] - voxel_grid_origin_x) / voxel_size);
    int pt_grid_y = round((pts[pt_idx * 3 + 1] - voxel_grid_origin_y) / voxel_size);
    int pt_grid_z = round((pts[pt_idx * 3 + 2] - voxel_grid_origin_z) / voxel_size);
    voxel_grid_occ[pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x] = 1.0f;
  }

  // Initialize TDF voxel grid
  float * voxel_grid_TDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  memset(voxel_grid_TDF, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  // Copy voxel grids to GPU memory
  float * gpu_voxel_grid_occ;
  float * gpu_voxel_grid_TDF;
  cudaMalloc(&gpu_voxel_grid_occ, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  cudaMalloc(&gpu_voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  marvin::checkCUDA(__LINE__, cudaGetLastError());
  cudaMemcpy(gpu_voxel_grid_occ, voxel_grid_occ, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_voxel_grid_TDF, voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  marvin::checkCUDA(__LINE__, cudaGetLastError());

  int CUDA_NUM_LOOPS = (int)ceil((float)(voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z) / (float)(CUDA_NUM_THREADS * CUDA_MAX_NUM_BLOCKS));

  for (int CUDA_LOOP_IDX = 0; CUDA_LOOP_IDX < CUDA_NUM_LOOPS; ++CUDA_LOOP_IDX) {
    ComputeTDF <<< CUDA_MAX_NUM_BLOCKS, CUDA_NUM_THREADS >>>(CUDA_LOOP_IDX, gpu_voxel_grid_occ, gpu_voxel_grid_TDF,
        voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
        voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
        voxel_size, trunc_margin);
  }

  // Load TDF voxel grid from GPU to CPU memory
  cudaMemcpy(voxel_grid_TDF, gpu_voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  marvin::checkCUDA(__LINE__, cudaGetLastError());

  // Compute random surface keypoints in point cloud coordinates and voxel grid coordinates
  std::cout << "Finding random surface keypoints..." << std::endl;
  int num_keypts = 50 * 10;
  float * keypts = new float[num_keypts * 3];
  float * keypts_grid = new float[num_keypts * 3];
  for (int keypt_idx = 0; keypt_idx < num_keypts; ++keypt_idx) {
    int rand_idx = (int)(GetRandomFloat(0.0f, (float)num_pts));
    keypts[keypt_idx * 3 + 0] = pts[rand_idx * 3 + 0];
    keypts[keypt_idx * 3 + 1] = pts[rand_idx * 3 + 1];
    keypts[keypt_idx * 3 + 2] = pts[rand_idx * 3 + 2];
    keypts_grid[keypt_idx * 3 + 0] = round((pts[rand_idx * 3 + 0] - voxel_grid_origin_x) / voxel_size);
    keypts_grid[keypt_idx * 3 + 1] = round((pts[rand_idx * 3 + 1] - voxel_grid_origin_y) / voxel_size);
    keypts_grid[keypt_idx * 3 + 2] = round((pts[rand_idx * 3 + 2] - voxel_grid_origin_z) / voxel_size);
  }

  // Start Marvin network
  marvin::Net convnet("3dmatch-net-test.json");
  convnet.Malloc(marvin::Testing);
  convnet.loadWeights("3dmatch-weights-snapshot-137000.marvin");
  marvin::Response * rData;
  marvin::Response * rFeat;
  rData = convnet.getResponse("data");
  rFeat = convnet.getResponse("feat");
  std::cout << "3DMatch network architecture successfully loaded into Marvin!" << std::endl;

  // Run forward passes with Marvin to get 3DMatch descriptors for each keypoint
  int batch_size = 50;
  int desc_size = 512;
  StorageT * batch_TDF = new StorageT[batch_size * 30 * 30 * 30];
  float * desc_3dmatch = new float[num_keypts * desc_size];
  std::cout << "Computing 3DMatch descriptors for " << num_keypts << " keypoints..." << std::endl;
  for (int batch_idx = 0; batch_idx < (num_keypts / batch_size); ++batch_idx) {
    for (int keypt_idx = batch_idx * batch_size; keypt_idx < (batch_idx + 1) * batch_size; ++keypt_idx) {

      int batch_keypt_idx = keypt_idx - batch_idx * batch_size;
      float keypt_grid_x = keypts_grid[keypt_idx * 3 + 0];
      float keypt_grid_y = keypts_grid[keypt_idx * 3 + 1];
      float keypt_grid_z = keypts_grid[keypt_idx * 3 + 2];

      // std::cout << keypt_idx << " " << batch_keypt_idx << std::endl;
      // std::cout << "    " << keypt_grid_x << " " << keypt_grid_y << " " << keypt_grid_z << std::endl;

      // Get local TDF around keypoint
      StorageT * local_voxel_grid_TDF = new StorageT[30 * 30 * 30];
      int local_voxel_idx = 0;
      for (int z = keypt_grid_z - 15; z < keypt_grid_z + 15; ++z)
        for (int y = keypt_grid_y - 15; y < keypt_grid_y + 15; ++y)
          for (int x = keypt_grid_x - 15; x < keypt_grid_x + 15; ++x) {
            local_voxel_grid_TDF[local_voxel_idx] = CPUCompute2StorageT(voxel_grid_TDF[z * voxel_grid_dim_x * voxel_grid_dim_y + y * voxel_grid_dim_x + x]);
            local_voxel_idx++;
          }
      for (int voxel_idx = 0; voxel_idx < 30 * 30 * 30; ++voxel_idx)
        batch_TDF[batch_keypt_idx * 30 * 30 * 30 + voxel_idx] = local_voxel_grid_TDF[voxel_idx];
      delete [] local_voxel_grid_TDF;
    }

    // Pass local TDF patches through Marvin
    cudaMemcpy(rData->dataGPU, batch_TDF, rData->numBytes(), cudaMemcpyHostToDevice);
    marvin::checkCUDA(__LINE__, cudaGetLastError());
    convnet.forward();

    // Copy descriptor vectors from GPU to CPU memory
    StorageT * desc_vecs = new StorageT[batch_size * desc_size];
    cudaMemcpy(desc_vecs, rFeat->dataGPU, rFeat->numBytes(), cudaMemcpyDeviceToHost);
    marvin::checkCUDA(__LINE__, cudaGetLastError());

    for (int desc_val_idx = 0; desc_val_idx < batch_size * desc_size; ++desc_val_idx)
      desc_3dmatch[batch_idx * batch_size * desc_size + desc_val_idx] = CPUStorage2ComputeT(desc_vecs[desc_val_idx]);

    delete [] desc_vecs;
  }

  // Save keypoints as binary file (Nx3 float array, row-major order)
  std::cout << "Saving keypoints to disk (keypts.bin)..." << std::endl;
  std::string keypts_saveto_path = out_prefix_filename + ".keypts.bin";
  std::ofstream keypts_out_file(keypts_saveto_path, std::ios::binary | std::ios::out);
  float num_keyptsf = (float) num_keypts;
  keypts_out_file.write((char*)&num_keyptsf, sizeof(float));
  for (int keypt_val_idx = 0; keypt_val_idx < num_keypts * 3; ++keypt_val_idx)
    keypts_out_file.write((char*)&keypts[keypt_val_idx], sizeof(float));
  keypts_out_file.close();

  // Save 3DMatch descriptors as binary file (Nx512 float array, row-major order)
  std::cout << "Saving 3DMatch descriptors to disk (desc.3dmatch.bin)..." << std::endl;
  std::string desc_saveto_path = out_prefix_filename + ".desc.3dmatch.bin";
  std::ofstream desc_out_file(desc_saveto_path, std::ios::binary | std::ios::out);
  float desc_sizef = (float) desc_size;
  desc_out_file.write((char*)&num_keyptsf, sizeof(float));
  desc_out_file.write((char*)&desc_sizef, sizeof(float));
  for (int desc_val_idx = 0; desc_val_idx < num_keypts * desc_size; ++desc_val_idx)
    desc_out_file.write((char*)&desc_3dmatch[desc_val_idx], sizeof(float));
  desc_out_file.close();

  // // Save TDF voxel grid and its parameters to disk as binary file (float array)
  // std::cout << "Saving TDF voxel grid values to disk (tdf.bin)..." << std::endl;
  // std::string voxel_grid_saveto_path = "tdf.bin";
  // std::ofstream tdf_out_file(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
  // float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
  // float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
  // float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
  // tdf_out_file.write((char*)&voxel_grid_dim_xf, sizeof(float));
  // tdf_out_file.write((char*)&voxel_grid_dim_yf, sizeof(float));
  // tdf_out_file.write((char*)&voxel_grid_dim_zf, sizeof(float));
  // tdf_out_file.write((char*)&voxel_grid_origin_x, sizeof(float));
  // tdf_out_file.write((char*)&voxel_grid_origin_y, sizeof(float));
  // tdf_out_file.write((char*)&voxel_grid_origin_z, sizeof(float));
  // tdf_out_file.write((char*)&voxel_size, sizeof(float));
  // tdf_out_file.write((char*)&trunc_margin, sizeof(float));
  // for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
  //   tdf_out_file.write((char*)&voxel_grid_TDF[i], sizeof(float));
  // tdf_out_file.close();

  return 0;
}


