// ---------------------------------------------------------
// Copyright (c) 2016, Andy Zeng
//
// This file is part of the 3DMatch Toolbox and is available
// under the terms of the Simplified BSD License provided in
// LICENSE. Please retain this notice and LICENSE if you use
// this file (or any portion of it) in your project.
// ---------------------------------------------------------

#include "utils.hpp"

// CUDA kernel function: for each voxel center of the voxel grid, find nearest neighbor distance to the surface point cloud
__global__
void Get1NNSearchDist(int num_surface_pts,
                      int voxel_grid_dimX, int voxel_grid_dimY, int voxel_grid_dimZ,
                      float min_bound_3d_x, float min_bound_3d_y, float min_bound_3d_z,
                      float max_bound_3d_x, float max_bound_3d_y, float max_bound_3d_z,
                      float * GPU_surface_pts_3d, float * GPU_closest_pt_dists) {

  int pt_idx = (blockIdx.x * CUDA_NUM_THREADS + threadIdx.x);
  if (pt_idx >= voxel_grid_dimX * voxel_grid_dimY * voxel_grid_dimZ)
    return;

  // Get voxel grid coordinates of current thread
  int z = floorf(pt_idx / (voxel_grid_dimX * voxel_grid_dimY)); //
  int y = floorf((pt_idx - (z * voxel_grid_dimX * voxel_grid_dimY)) / voxel_grid_dimX);
  int x = pt_idx - (z * voxel_grid_dimX * voxel_grid_dimY) - (y * voxel_grid_dimX);

  float voxel_grid_unit_x = (max_bound_3d_x - min_bound_3d_x) / (float)voxel_grid_dimX;
  float voxel_grid_unit_y = (max_bound_3d_y - min_bound_3d_y) / (float)voxel_grid_dimY;
  float voxel_grid_unit_z = (max_bound_3d_z - min_bound_3d_z) / (float)voxel_grid_dimZ;

  // Convert from voxel grid coordinates to camera coordinates
  float pt_cam_x = ((float)x + 0.5f) * voxel_grid_unit_x + min_bound_3d_x;
  float pt_cam_y = ((float)y + 0.5f) * voxel_grid_unit_y + min_bound_3d_y;
  float pt_cam_z = ((float)z + 0.5f) * voxel_grid_unit_z + min_bound_3d_z;

  // Compute distance from voxel center to closest surface point
  float closest_dist = -1.0f;
  for (int i = 0; i < num_surface_pts; ++i) {
    float query_pt_x = GPU_surface_pts_3d[3 * i + 0];
    float query_pt_y = GPU_surface_pts_3d[3 * i + 1];
    float query_pt_z = GPU_surface_pts_3d[3 * i + 2];
    float query_dist = sqrtf((pt_cam_x - query_pt_x) * (pt_cam_x - query_pt_x) +
                             (pt_cam_y - query_pt_y) * (pt_cam_y - query_pt_y) +
                             (pt_cam_z - query_pt_z) * (pt_cam_z - query_pt_z));
    if (closest_dist == -1.0f || query_dist < closest_dist)
      closest_dist = query_dist;
  }
  GPU_closest_pt_dists[pt_idx] = closest_dist;
}

// 3DMatch data layer for Marvin
template <class T>
class MatchDataLayer : public DataLayer {
  std::future<void> lock;

  std::vector<StorageT*> data_CPU;
  std::vector<StorageT*> data_GPU;
  std::vector<StorageT*> label_CPU;
  std::vector<StorageT*> label_GPU;

public:

  std::string data_path;

  int batch_size;

  // Pre-defined parameters
  int im_height = 480;
  int im_width = 640;
  int volume_dim = 30;

  int numofitems() { return 0; };

  void init() {
    train_me = true;
    std::cout << "MatchDataLayer: " << std::endl;
    data_CPU.resize(3);
    data_GPU.resize(3);
    label_CPU.resize(2);
    label_GPU.resize(2);

    // Compute batch data sizes
    std::vector<int> data_dim;
    data_dim.push_back(batch_size); data_dim.push_back(1); data_dim.push_back(volume_dim); data_dim.push_back(volume_dim); data_dim.push_back(volume_dim);
    data_CPU[0]  = new StorageT[numel(data_dim)];
    data_CPU[1]  = new StorageT[numel(data_dim)];
    data_CPU[2]  = new StorageT[numel(data_dim)];

    // Compute batch label sizes
    std::vector<int> label_dim;
    label_dim.push_back(batch_size); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1);
    label_CPU[0] = new StorageT[numel(label_dim)];
    label_CPU[1] = new StorageT[numel(label_dim)];

  };

  MatchDataLayer(std::string name_, Phase phase_, std::string data_path_, int batch_size_):
    DataLayer(name_), data_path(data_path_), batch_size(batch_size_) {
    phase = phase_;
    init();
  };

  MatchDataLayer(JSON* json) {
    SetOrDie(json, name)
    SetValue(json, phase, Training)
    SetOrDie(json, data_path)
    SetOrDie(json, batch_size)
    init();
  };

  ~MatchDataLayer() {
    if (lock.valid()) lock.wait();
    for (int i = 0; i < data_CPU.size(); ++i)
      if (data_CPU[i] != NULL) delete [] data_CPU[i];
    for (int i = 0; i < label_CPU.size(); ++i)
      if (label_CPU[i] != NULL) delete [] label_CPU[i];
    for (int i = 0; i < data_GPU.size(); ++i)
      if (data_GPU[i] != NULL) checkCUDA(__LINE__, cudaFree(data_GPU[i]));
    for (int i = 0; i < label_GPU.size(); ++i)
      if (label_GPU[i] != NULL) checkCUDA(__LINE__, cudaFree(label_GPU[i]));
  };

  void shuffle() {};

  // Given a depth map and the pixel coordinates of a point p, compute the local volumetric voxel grid of TDF values around p
  void GetLocalPointvoxel_grid_TDF(int pix_x, int pix_y, float * cam_K, float * depth_im_p1, int im_height, int im_width, float * voxel_grid_TDF, int voxel_grid_dim, float voxel_size, float trunc_margin) {

    // Project pixel location to 3D point in camera coordinates
    float pt_cam_z = depth_im_p1[pix_y * im_width + pix_x];
    float pt_cam_x = ((float)(pix_x) + 0.5f - cam_K[0 * 3 + 2]) * pt_cam_z / cam_K[0 * 3 + 0];
    float pt_cam_y = ((float)(pix_y) + 0.5f - cam_K[1 * 3 + 2]) * pt_cam_z / cam_K[1 * 3 + 1];

    // Get pixel bounding box of local volume
    float loose_box_radius = ((float)voxel_grid_dim / 2.0f + 3.0f) * voxel_size; // Bounding box margin size: 3 voxels
    float bounds_3d_x[2] = {pt_cam_x + loose_box_radius, pt_cam_x - loose_box_radius};
    float bounds_3d_y[2] = {pt_cam_y + loose_box_radius, pt_cam_y - loose_box_radius};
    float bounds_3d_z[2] = {pt_cam_z + loose_box_radius, pt_cam_z - loose_box_radius};
    float bbox_pts_3d[8][3] = {{bounds_3d_x[0], bounds_3d_y[0], bounds_3d_z[0]},
      {bounds_3d_x[0], bounds_3d_y[0], bounds_3d_z[1]},
      {bounds_3d_x[0], bounds_3d_y[1], bounds_3d_z[0]},
      {bounds_3d_x[0], bounds_3d_y[1], bounds_3d_z[1]},
      {bounds_3d_x[1], bounds_3d_y[0], bounds_3d_z[0]},
      {bounds_3d_x[1], bounds_3d_y[0], bounds_3d_z[1]},
      {bounds_3d_x[1], bounds_3d_y[1], bounds_3d_z[0]},
      {bounds_3d_x[1], bounds_3d_y[1], bounds_3d_z[1]}
    };
    float min_bounds_2d[2] = {(float)im_width, (float)im_height}; // x,y
    float max_bounds_2d[2] = {0};
    for (int i = 0; i < 8; ++i) {
      float tmp_pix_x = std::round((bbox_pts_3d[i][0] * cam_K[0 * 3 + 0] / bbox_pts_3d[i][2]) + cam_K[0 * 3 + 2] - 0.5f);
      float tmp_pix_y = std::round((bbox_pts_3d[i][1] * cam_K[1 * 3 + 1] / bbox_pts_3d[i][2]) + cam_K[1 * 3 + 2] - 0.5f);
      min_bounds_2d[0] = std::min(tmp_pix_x, min_bounds_2d[0]);
      min_bounds_2d[1] = std::min(tmp_pix_y, min_bounds_2d[1]);
      max_bounds_2d[0] = std::max(tmp_pix_x, max_bounds_2d[0]);
      max_bounds_2d[1] = std::max(tmp_pix_y, max_bounds_2d[1]);
    }
    min_bounds_2d[0] = std::max(min_bounds_2d[0], 0.0f);
    min_bounds_2d[1] = std::max(min_bounds_2d[1], 0.0f);
    max_bounds_2d[0] = std::min(max_bounds_2d[0], (float)im_width - 1.0f);
    max_bounds_2d[1] = std::min(max_bounds_2d[1], (float)im_height - 1.0f);

    // Project pixels in image bounding box to 3D
    int num_local_region_pts = (max_bounds_2d[0] - min_bounds_2d[0] + 1) * (max_bounds_2d[1] - min_bounds_2d[1] + 1);
    // std::cout << num_local_region_pts << std::endl;
    float * num_local_region_pts_3d = new float[3 * num_local_region_pts];
    int num_local_region_pts_3dIdx = 0;
    for (int y = min_bounds_2d[1]; y <= max_bounds_2d[1]; ++y)
      for (int x = min_bounds_2d[0]; x <= max_bounds_2d[0]; ++x) {
        float tmp_pt_cam_z = depth_im_p1[y * im_width + x];
        float tmp_pt_cam_x = ((float)x + 0.5f - cam_K[0 * 3 + 2]) * tmp_pt_cam_z / cam_K[0 * 3 + 0];
        float tmp_pt_cam_y = ((float)y + 0.5f - cam_K[1 * 3 + 2]) * tmp_pt_cam_z / cam_K[1 * 3 + 1];
        num_local_region_pts_3d[3 * num_local_region_pts_3dIdx + 0] = tmp_pt_cam_x;
        num_local_region_pts_3d[3 * num_local_region_pts_3dIdx + 1] = tmp_pt_cam_y;
        num_local_region_pts_3d[3 * num_local_region_pts_3dIdx + 2] = tmp_pt_cam_z;
        num_local_region_pts_3dIdx++;
      }

    // FILE *fp = fopen("test.txt", "w");
    // for (int i = 0; i < num_local_region_pts; ++i) {
    //     std::cout << num_local_region_pts_3d[3 * i + 0] << " " << num_local_region_pts_3d[3 * i + 1] << " " << num_local_region_pts_3d[3 * i + 2] << std::endl;
    //     float tmpx = num_local_region_pts_3d[3 * i + 0];
    //     float tmpy = num_local_region_pts_3d[3 * i + 1];
    //     float tmpz = num_local_region_pts_3d[3 * i + 2];
    //     int iret = fprintf(fp, "%f %f %f\n",tmpx,tmpy,tmpz);
    // }
    // for (int i = 0; i < 8; ++i) {
    //     float tmpx = bbox_pts_3d[i][0];
    //     float tmpy = bbox_pts_3d[i][1];
    //     float tmpz = bbox_pts_3d[i][2];
    //     int iret = fprintf(fp, "%f %f %f\n",tmpx,tmpy,tmpz);
    // }
    // fclose(fp);

    // Prepare GPU variables
    int voxel_grid_dimX = voxel_grid_dim;
    int voxel_grid_dimY = voxel_grid_dim;
    int voxel_grid_dimZ = voxel_grid_dim;
    int num_grid_pts = voxel_grid_dimX * voxel_grid_dimY * voxel_grid_dimZ;
    float * closest_pt_dists = new float[num_grid_pts];
    float * GPU_local_region_pts_3d;
    float * GPU_closest_pt_dists;
    cudaMalloc(&GPU_local_region_pts_3d, 3 * num_local_region_pts * sizeof(float));
    cudaMalloc(&GPU_closest_pt_dists, num_grid_pts * sizeof(float));
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(GPU_local_region_pts_3d, num_local_region_pts_3d, 3 * num_local_region_pts * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    // Compute xyz range of 3d local voxel grid in camera coordinates
    float tight_box_radius = ((float)voxel_grid_dim / 2.0f) * voxel_size;
    float min_bound_3d_x = pt_cam_x - tight_box_radius;
    float max_bound_3d_x = pt_cam_x + tight_box_radius;
    float min_bound_3d_y = pt_cam_y - tight_box_radius;
    float max_bound_3d_y = pt_cam_y + tight_box_radius;
    float min_bound_3d_z = pt_cam_z - tight_box_radius;
    float max_bound_3d_z = pt_cam_z + tight_box_radius;

    // Compute voxel grid of TDF values (use CUDA GPU kernel function)
    int TMP_CUDA_NUM_BLOCKS = std::ceil((float)(voxel_grid_dim * voxel_grid_dim * voxel_grid_dim) / (float)CUDA_NUM_THREADS);
    Get1NNSearchDist <<< TMP_CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>> (num_local_region_pts,
        voxel_grid_dimX, voxel_grid_dimY, voxel_grid_dimZ,
        min_bound_3d_x, min_bound_3d_y, min_bound_3d_z,
        max_bound_3d_x, max_bound_3d_y, max_bound_3d_z,
        GPU_local_region_pts_3d, GPU_closest_pt_dists);
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(closest_pt_dists, GPU_closest_pt_dists, num_grid_pts * sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDA(__LINE__, cudaGetLastError());

    // Apply truncation, normalize between 0 and 1, and flip so 1 is near surface and 0 is far away from surface
    for (int i = 0; i < num_grid_pts; ++i)
      voxel_grid_TDF[i] = 1.0f - (std::min(closest_pt_dists[i], trunc_margin) / trunc_margin);

    // std::cout << pt_cam_x << " " << pt_cam_y << " " << pt_cam_z << std::endl;
    // std::cout << pix_x << " " << pix_y << std::endl;
    // std::cout << min_bounds_2d[0] << " " << min_bounds_2d[1] << " " << max_bounds_2d[0] << " " << max_bounds_2d[1] << std::endl;

    delete [] num_local_region_pts_3d;
    delete [] closest_pt_dists;
    checkCUDA(__LINE__, cudaFree(GPU_local_region_pts_3d));
    checkCUDA(__LINE__, cudaFree(GPU_closest_pt_dists));
  }

  void prefetch() {

    checkCUDA(__LINE__, cudaSetDevice(GPU));

    // Naming convention: p1 and p2 are matches, p1 and p3 are non-matches
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {

      float * depth_im_p1 = new float[im_height * im_width];
      float * depth_im_p2 = new float[im_height * im_width];
      float * depth_im_p3 = new float[im_height * im_width];

      float cam_K_p1[9];
      float cam_K_p2[9];
      float cam_K_p3[9];

      std::string file_prefix_p1;
      int p1_pix_x;
      int p1_pix_y;
      std::string file_prefix_p2;
      int p2_pix_x;
      int p2_pix_y;
      std::string file_prefix_p3;
      int p3_pix_x;
      int p3_pix_y;

      // Find positive correspondence p1 and p2
      while (true) {

        // Find a random scene
        std::vector<std::string> scene_list;
        GetFilesInDirectory(data_path, scene_list, "");
        int rand_scene_idx = (int)round(GetRandomFloat(0.0f, (float)(scene_list.size() - 1)));
        std::string scene_name = scene_list[rand_scene_idx];

        // Find a random sequence
        std::vector<std::string> seq_list;
        GetFilesInDirectory(data_path + "/" + scene_name, seq_list, "seq");
        int rand_seq_idx = (int)round(GetRandomFloat(0.0f, (float)(seq_list.size() - 1)));
        std::string seq_name = seq_list[rand_seq_idx];

        // Find a random frame
        std::vector<std::string> frame_list_p1;
        GetFilesInDirectory(data_path + "/" + scene_name + "/" + seq_name, frame_list_p1, ".depth.png");
        int rand_frame_idx = (int)round(GetRandomFloat(0.0f, (float)(frame_list_p1.size() - 1)));
        std::string frame_name_p1 = frame_list_p1[rand_frame_idx];

        // Pick a random pixel
        float rand_pix_x = round(GetRandomFloat(0.0f, (float)(im_width - 1)));
        float rand_pix_y = round(GetRandomFloat(0.0f, (float)(im_height - 1)));

        // Read depth image
        ReadDepth(data_path + "/" + scene_name + "/" + seq_name + "/" + frame_name_p1, im_height, im_width, depth_im_p1);

        // Get camera intrinsics
        std::string cam_K_file_p1 = data_path + "/" + scene_name + "/camera-intrinsics.txt";
        std::vector<float> cam_K_vec_p1 = LoadMatrixFromFile(cam_K_file_p1, 3, 3);
        std::copy(cam_K_vec_p1.begin(), cam_K_vec_p1.end(), cam_K_p1);

        // Get depth at pixel. If it is zero, resample
        float depth_val_p1 = depth_im_p1[(int)rand_pix_y * im_width + (int)rand_pix_x];
        if (depth_val_p1 == 0)
          continue;

        // Save p1
        file_prefix_p1 = data_path + "/" + scene_name + "/" + seq_name + "/" + frame_name_p1.substr(0, frame_name_p1.length() - 10);
        p1_pix_x = (int)rand_pix_x;
        p1_pix_y = (int)rand_pix_y;

        // Project point to 3D camera coordinates
        float p1_cam_z = depth_val_p1;
        float p1_cam_x = (rand_pix_x + 0.5f - cam_K_p1[2]) * p1_cam_z / cam_K_p1[0];
        float p1_cam_y = (rand_pix_y + 0.5f - cam_K_p1[5]) * p1_cam_z / cam_K_p1[4];

        // Get extrinsics of base frame
        std::string cam2world_file_p1 = data_path + "/" + scene_name + "/" + seq_name + "/" + frame_name_p1.substr(0, frame_name_p1.length() - 10) + ".pose.txt";
        std::vector<float> cam2world_p1 = LoadMatrixFromFile(cam2world_file_p1, 4, 4);

        // Convert 3D point to world coordinates
        float p1_world_x = cam2world_p1[0] * p1_cam_x + cam2world_p1[1] * p1_cam_y + cam2world_p1[2] * p1_cam_z + cam2world_p1[3];
        float p1_world_y = cam2world_p1[4] * p1_cam_x + cam2world_p1[5] * p1_cam_y + cam2world_p1[6] * p1_cam_z + cam2world_p1[7];
        float p1_world_z = cam2world_p1[8] * p1_cam_x + cam2world_p1[9] * p1_cam_y + cam2world_p1[10] * p1_cam_z + cam2world_p1[11];

        // Get list of frames from the scene with all valid positive correspondences
        std::vector<std::string> valid_frame_list_p2;
        std::vector<float> valid_p2_pix_x;
        std::vector<float> valid_p2_pix_y;
        std::vector<float> valid_p2_depth;
        for (int seq_idx = 0; seq_idx < seq_list.size(); seq_idx++) {
          std::string curr_seq_name = seq_list[seq_idx];

          std::vector<std::string> curr_seq_frame_list;
          GetFilesInDirectory(data_path + "/" + scene_name + "/" + curr_seq_name, curr_seq_frame_list, ".pose.txt");

          // std::cout << curr_seq_frame_list.size() << std::endl;
          for (int frame_idx = 0; frame_idx < curr_seq_frame_list.size(); frame_idx++) {

            // Get extrinsics of current frame
            std::string curr_seq_frame_cam2world_file = data_path + "/" + scene_name + "/" + curr_seq_name + "/" + curr_seq_frame_list[frame_idx];
            std::vector<float> curr_seq_frame_cam2world = LoadMatrixFromFile(curr_seq_frame_cam2world_file, 4, 4);

            // Get camera intrinsics of current frame
            std::string cam_K_p2_file = data_path + "/" + scene_name + "/camera-intrinsics.txt";
            std::vector<float> cam_K_p2_vec = LoadMatrixFromFile(cam_K_p2_file, 3, 3);
            std::copy(cam_K_p2_vec.begin(), cam_K_p2_vec.end(), cam_K_p2);

            // Project p1 from world coordinates to image pixel coordinates of current frame
            float p2_cam_xt = p1_world_x - curr_seq_frame_cam2world[3];
            float p2_cam_yt = p1_world_y - curr_seq_frame_cam2world[7];
            float p2_cam_zt = p1_world_z - curr_seq_frame_cam2world[11];
            float p2_cam_x = curr_seq_frame_cam2world[0] * p2_cam_xt + curr_seq_frame_cam2world[4] * p2_cam_yt + curr_seq_frame_cam2world[8] * p2_cam_zt;
            float p2_cam_y = curr_seq_frame_cam2world[1] * p2_cam_xt + curr_seq_frame_cam2world[5] * p2_cam_yt + curr_seq_frame_cam2world[9] * p2_cam_zt;
            float p2_cam_z = curr_seq_frame_cam2world[2] * p2_cam_xt + curr_seq_frame_cam2world[6] * p2_cam_yt + curr_seq_frame_cam2world[10] * p2_cam_zt;
            float proj_p2_pix_x = round((cam_K_p2[0] * (p2_cam_x / p2_cam_z) + cam_K_p2[2]) - 0.5f);
            float proj_p2_pix_y = round((cam_K_p2[4] * (p2_cam_y / p2_cam_z) + cam_K_p2[5]) - 0.5f);

            // Check if pixel coordinates are within image bounds
            if ((int) proj_p2_pix_x < 0 || (int) proj_p2_pix_x >= im_width || (int) proj_p2_pix_y < 0 || (int) proj_p2_pix_y >= im_height)
              continue;

            // Check if camera location of current frame is at least 1m away from camera location of frame of p1
            if (sqrtf((curr_seq_frame_cam2world[3] - cam2world_p1[3]) * (curr_seq_frame_cam2world[3] - cam2world_p1[3]) + 
                      (curr_seq_frame_cam2world[7] - cam2world_p1[7]) * (curr_seq_frame_cam2world[7] - cam2world_p1[7]) + 
                      (curr_seq_frame_cam2world[11] - cam2world_p1[11]) * (curr_seq_frame_cam2world[11] - cam2world_p1[11])) < 1.0f)
              continue;

            valid_frame_list_p2.push_back(curr_seq_frame_cam2world_file.substr(0, curr_seq_frame_cam2world_file.length() - 9));
            valid_p2_pix_x.push_back(proj_p2_pix_x);
            valid_p2_pix_y.push_back(proj_p2_pix_y);
            valid_p2_depth.push_back(p2_cam_z);
          }
        }

        if (valid_frame_list_p2.size() == 0)
          continue;

        // Pick a random positive correspondence (max: 100 tries)
        std::string frame_name_p2;
        for (int try_idx = 0; try_idx < 100; try_idx++) {
          rand_frame_idx = (int)round(GetRandomFloat(0.0f, (float)(valid_frame_list_p2.size() - 1)));
          frame_name_p2 = valid_frame_list_p2[rand_frame_idx];

          // Check that positive correspondence has depth in other frame
          std::string depth_im_file_p2 = frame_name_p2 + ".depth.png";
          ReadDepth(depth_im_file_p2, im_height, im_width, depth_im_p2);
          float depth_val_p2 = depth_im_p2[(int)(valid_p2_pix_y[rand_frame_idx]) * im_width + (int)(valid_p2_pix_x[rand_frame_idx])];
          if (depth_val_p2 == 0)
            continue;

          // Check that positive correspondence in world coordinates is close enough
          if (abs(valid_p2_depth[rand_frame_idx] - depth_val_p2) > 0.03)
            continue;

          break;
        }

        // Set positive correspondence point as p2
        file_prefix_p2 = frame_name_p2;
        p2_pix_x = (int)(valid_p2_pix_x[rand_frame_idx]);
        p2_pix_y = (int)(valid_p2_pix_y[rand_frame_idx]);
        break;
      }

      // Find a negative correspondence p3
      while (true) {

        // Find a random scene
        std::vector<std::string> scene_list;
        GetFilesInDirectory(data_path, scene_list, "");
        int rand_scene_idx = (int)round(GetRandomFloat(0.0f, (float)(scene_list.size() - 1)));
        std::string scene_name = scene_list[rand_scene_idx];

        // Find a random sequence
        std::vector<std::string> seq_list;
        GetFilesInDirectory(data_path + "/" + scene_name, seq_list, "seq");
        int rand_seq_idx = (int)round(GetRandomFloat(0.0f, (float)(seq_list.size() - 1)));
        std::string seq_name = seq_list[rand_seq_idx];

        // Find a random frame
        std::vector<std::string> frame_list_p3;
        GetFilesInDirectory(data_path + "/" + scene_name + "/" + seq_name, frame_list_p3, ".depth.png");
        int rand_frame_idx = (int)round(GetRandomFloat(0.0f, (float)(frame_list_p3.size() - 1)));
        std::string frame_name_p3 = frame_list_p3[rand_frame_idx];

        // Pick a random pixel
        float rand_p3_pix_x = round(GetRandomFloat(0.0f, (float)(im_width - 1)));
        float rand_p3_pix_y = round(GetRandomFloat(0.0f, (float)(im_height - 1)));

        // Read depth image
        ReadDepth(data_path + "/" + scene_name + "/" + seq_name + "/" + frame_name_p3, im_height, im_width, depth_im_p3);

        // Get camera intrinsics
        std::string cam_K_p3_file = data_path + "/" + scene_name + "/camera-intrinsics.txt";
        std::vector<float> cam_K_p3_vec = LoadMatrixFromFile(cam_K_p3_file, 3, 3);
        std::copy(cam_K_p3_vec.begin(), cam_K_p3_vec.end(), cam_K_p3);

        // Get depth at pixel. If it is zero, resample
        float depth_val_p3 = depth_im_p3[(int)rand_p3_pix_y * im_width + (int)rand_p3_pix_x];
        if (depth_val_p3 == 0)
          continue;

        // Set negative correspondence point as p3
        file_prefix_p3 = data_path + "/" + scene_name + "/" + seq_name + "/" + frame_name_p3.substr(0, frame_name_p3.length() - 10);
        p3_pix_x = (int)rand_p3_pix_x;
        p3_pix_y = (int)rand_p3_pix_y;
        break;
      }

      // For debugging
      // std::cout << file_prefix_p1 << " " << p1_pix_x << " " << p1_pix_y << std::endl;
      // std::cout << file_prefix_p2 << " " << p2_pix_x << " " << p2_pix_y << std::endl;
      // std::cout << file_prefix_p3 << " " << p3_pix_x << " " << p3_pix_y << std::endl << std::endl;

      std::string depth_im_file_p1 = file_prefix_p1 + ".depth.png";
      std::string cam_pose_file_p1 = file_prefix_p1 + ".pose.txt";
      std::string depth_im_file_p2 = file_prefix_p2 + ".depth.png";
      std::string cam_pose_file_p2 = file_prefix_p2 + ".pose.txt";
      std::string depth_im_file_p3 = file_prefix_p3 + ".depth.png";
      std::string cam_pose_file_p3 = file_prefix_p3 + ".pose.txt";

      // Convert depth image to row-major array
      ReadDepth(depth_im_file_p1, im_height, im_width, depth_im_p1);

      // TDF voxel grid parameters
      int voxel_grid_dim = 30; // In voxels
      float voxel_size = 0.01f; // In meters
      float trunc_margin = 5 * voxel_size;
      int num_grid_pts = voxel_grid_dim * voxel_grid_dim * voxel_grid_dim;

      // Compute TDF voxel grid around p1
      float * voxel_grid_TDF_p1 = new float[num_grid_pts];
      GetLocalPointvoxel_grid_TDF(p1_pix_x, p1_pix_y, cam_K_p1, depth_im_p1, im_height, im_width, voxel_grid_TDF_p1, voxel_grid_dim, voxel_size, trunc_margin);

      // For debugging
      // FILE *fp = fopen("debugBaseMatch.txt", "w");
      // int iret = fprintf(fp, "path:%s x:%d y:%d\n",baseFramePrefix.c_str(),corresBasePointX[match_idx],corresBasePointY[match_idx]);
      // for (int i = 0; i < num_grid_pts; ++i)
      //     iret = fprintf(fp, "%f\n",voxel_grid_TDF_p1[i]);
      // fclose(fp);

      // Compute TDF voxel grid around p2
      // float * depth_im_p2 = new float[im_height * im_width];
      ReadDepth(depth_im_file_p2, im_height, im_width, depth_im_p2);
      float * voxel_grid_TDF_p2 = new float[num_grid_pts];
      GetLocalPointvoxel_grid_TDF(p2_pix_x, p2_pix_y, cam_K_p2, depth_im_p2, im_height, im_width, voxel_grid_TDF_p2, voxel_grid_dim, voxel_size, trunc_margin);

      // fp = fopen("debugPosMatch.txt", "w");
      // iret = fprintf(fp, "path:%s x:%d y:%d\n",posFramePrefix.c_str(),corresPosPointX[match_idx],corresPosPointY[match_idx]);
      // for (int i = 0; i < num_grid_pts; ++i)
      //     iret = fprintf(fp, "%f\n",voxel_grid_TDF_p2[i]);
      // fclose(fp);

      // Compute TDF voxel grid around p3
      ReadDepth(depth_im_file_p3, im_height, im_width, depth_im_p3);
      float * voxel_grid_TDF_p3 = new float[num_grid_pts];
      GetLocalPointvoxel_grid_TDF(p3_pix_x, p3_pix_y, cam_K_p3, depth_im_p3, im_height, im_width, voxel_grid_TDF_p3, voxel_grid_dim, voxel_size, trunc_margin);

      // fp = fopen("debugNegMatch.txt", "w");
      // iret = fprintf(fp, "path:%s x:%d y:%d\n",negFramePrefix.c_str(),corresNegPointX[match_idx],corresNegPointY[match_idx]);
      // for (int i = 0; i < num_grid_pts; ++i)
      //     iret = fprintf(fp, "%f\n",voxel_grid_TDF_p3[i]);
      // fclose(fp);

      // Copy to data response
      checkCUDA(__LINE__, cudaMemcpy(&(data_GPU[0][batch_idx * num_grid_pts]), voxel_grid_TDF_p1, num_grid_pts * sizeofStorageT, cudaMemcpyHostToDevice));
      checkCUDA(__LINE__, cudaMemcpy(&(data_GPU[1][batch_idx * num_grid_pts]), voxel_grid_TDF_p2, num_grid_pts * sizeofStorageT, cudaMemcpyHostToDevice));
      checkCUDA(__LINE__, cudaMemcpy(&(data_GPU[2][batch_idx * num_grid_pts]), voxel_grid_TDF_p3, num_grid_pts * sizeofStorageT, cudaMemcpyHostToDevice));
      float pos_label = 1.0f;
      float neg_label = 0.0f;
      checkCUDA(__LINE__, cudaMemcpy(&(label_GPU[0][batch_idx]), &pos_label, sizeofStorageT, cudaMemcpyHostToDevice));
      checkCUDA(__LINE__, cudaMemcpy(&(label_GPU[1][batch_idx]), &neg_label, sizeofStorageT, cudaMemcpyHostToDevice));

      // Clear memory
      delete [] depth_im_p1;
      delete [] voxel_grid_TDF_p1;
      delete [] depth_im_p2;
      delete [] voxel_grid_TDF_p2;
      delete [] depth_im_p3;
      delete [] voxel_grid_TDF_p3;
    }

  };

  void forward(Phase phase_) {
    lock.wait();
    std::swap(out[0]->dataGPU, data_GPU[0]);
    std::swap(out[1]->dataGPU, data_GPU[1]);
    std::swap(out[2]->dataGPU, data_GPU[2]);
    std::swap(out[3]->dataGPU, label_GPU[0]);
    std::swap(out[4]->dataGPU, label_GPU[1]);
    lock = std::async(std::launch::async, &MatchDataLayer<T>::prefetch, this);
    // prefetch();
  };


  size_t Malloc(Phase phase_) {
    if (phase == Training && phase_ == Testing) return 0;
    if (out.size() != 5) {
      std::cout << "MatchDataLayer: incorrect # of out's" << std::endl;
      FatalError(__LINE__);
    }
    size_t memory_bytes = 0;
    std::cout << (train_me ? "* " : "  ");
    std::cout << name << std::endl;

    // CPU/GPU malloc data
    std::vector<int> data_dim;
    data_dim.push_back(batch_size); data_dim.push_back(1); data_dim.push_back(30); data_dim.push_back(30); data_dim.push_back(30);

    out[0]->need_diff = false;
    out[0]->receptive_field.resize(data_dim.size() - 2); fill_n(out[0]->receptive_field.begin(), data_dim.size() - 2, 1);
    out[0]->receptive_gap.resize(data_dim.size() - 2);   fill_n(out[0]->receptive_gap.begin(), data_dim.size() - 2, 1);
    out[0]->receptive_offset.resize(data_dim.size() - 2); fill_n(out[0]->receptive_offset.begin(), data_dim.size() - 2, 0);
    memory_bytes += out[0]->Malloc(data_dim);
    checkCUDA(__LINE__, cudaMalloc(&data_GPU[0], numel(data_dim) * sizeofStorageT) );
    memory_bytes += numel(data_dim) * sizeofStorageT;

    out[1]->need_diff = false;
    out[1]->receptive_field.resize(data_dim.size() - 2); fill_n(out[1]->receptive_field.begin(), data_dim.size() - 2, 1);
    out[1]->receptive_gap.resize(data_dim.size() - 2);   fill_n(out[1]->receptive_gap.begin(), data_dim.size() - 2, 1);
    out[1]->receptive_offset.resize(data_dim.size() - 2); fill_n(out[1]->receptive_offset.begin(), data_dim.size() - 2, 0);
    memory_bytes += out[1]->Malloc(data_dim);
    checkCUDA(__LINE__, cudaMalloc(&data_GPU[1], numel(data_dim) * sizeofStorageT) );
    memory_bytes += numel(data_dim) * sizeofStorageT;

    out[2]->need_diff = false;
    out[2]->receptive_field.resize(data_dim.size() - 2); fill_n(out[2]->receptive_field.begin(), data_dim.size() - 2, 1);
    out[2]->receptive_gap.resize(data_dim.size() - 2);   fill_n(out[2]->receptive_gap.begin(), data_dim.size() - 2, 1);
    out[2]->receptive_offset.resize(data_dim.size() - 2); fill_n(out[2]->receptive_offset.begin(), data_dim.size() - 2, 0);
    memory_bytes += out[2]->Malloc(data_dim);
    checkCUDA(__LINE__, cudaMalloc(&data_GPU[2], numel(data_dim) * sizeofStorageT) );
    memory_bytes += numel(data_dim) * sizeofStorageT;

    // CPU/GPU malloc labels
    std::vector<int> label_dim;
    label_dim.push_back(batch_size); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1);

    out[3]->need_diff = false;
    memory_bytes += out[3]->Malloc(label_dim);
    checkCUDA(__LINE__, cudaMalloc(&label_GPU[0], numel(label_dim) * sizeofStorageT) );
    memory_bytes += numel(label_dim) * sizeofStorageT;

    out[4]->need_diff = false;
    memory_bytes += out[4]->Malloc(label_dim);
    checkCUDA(__LINE__, cudaMalloc(&label_GPU[1], numel(label_dim) * sizeofStorageT) );
    memory_bytes += numel(label_dim) * sizeofStorageT;

    lock = std::async(std::launch::async, &MatchDataLayer<T>::prefetch, this);
    // prefetch();

    return memory_bytes;
  };
};
