// ---------------------------------------------------------
// Copyright (c) 2016, Andy Zeng
// 
// This file is part of the 3DMatch Toolbox and is available 
// under the terms of the Simplified BSD License provided in 
// LICENSE. Please retain this notice and LICENSE if you use 
// this file (or any portion of it) in your project.
// ---------------------------------------------------------

#include <random>
#include <algorithm>
#include <dirent.h>

// Return a random float between min and max
float GetRandomFloat(float min, float max) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(min, max - 0.0001);
  return dist(mt);
}

// Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
// Return the matrix as a float vector of the matrix in row-major order
std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N) {
  std::vector<float> matrix;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < M * N; i++) {
    float tmp;
    int iret = fscanf(fp, "%f", &tmp);
    matrix.push_back(tmp);
  }
  fclose(fp);
  return matrix;
}

// Save the names of all files in a folder into string vector file_list
// Only save the names containing search_string
void GetFilesInDirectory(const std::string &directory, std::vector<std::string> &file_list, const std::string &search_string) {
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (directory.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string filename(ent->d_name);
      if (filename.find(search_string) != std::string::npos && filename != "." && filename != "..")
        file_list.push_back(filename);
    }
    closedir (dir);
  } else
    perror ("Error: could not look into directory!");
}

// Trim white space from both sides of a string
static inline void TrimString(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float * depth) {
  cv::Mat depth_mat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
  if (depth_mat.empty()) {
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f;
      if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
        depth[r * W + c] = 0;
    }
}


// std::string GetRandomString(size_t str_len) {
//   auto rand_char = []() -> char {
//     const char char_set[] = "0123456789abcdefghijklmnopqrstuvwxyz";
//     return char_set[((int)std::floor(GetRandomFloat(0.0f, (float)sizeof(char_set) - 1)))];
//   };
//   std::string rand_str(str_len, 0);
//   std::generate_n(rand_str.begin(), str_len, rand_char);
//   return rand_str;
// }

// bool FileExists(const std::string &filename) {
//   std::ifstream file(filename);
//   return (!file.fail());
// }

// // 4x4 matrix inversion
// // Matrices are stored from left-to-right, top-to-bottom
// bool InvertMatrix(const float m[16], float invOut[16]) {
//   float inv[16], det;
//   int i;
//   inv[0] = m[5]  * m[10] * m[15] -
//            m[5]  * m[11] * m[14] -
//            m[9]  * m[6]  * m[15] +
//            m[9]  * m[7]  * m[14] +
//            m[13] * m[6]  * m[11] -
//            m[13] * m[7]  * m[10];

//   inv[4] = -m[4]  * m[10] * m[15] +
//            m[4]  * m[11] * m[14] +
//            m[8]  * m[6]  * m[15] -
//            m[8]  * m[7]  * m[14] -
//            m[12] * m[6]  * m[11] +
//            m[12] * m[7]  * m[10];

//   inv[8] = m[4]  * m[9] * m[15] -
//            m[4]  * m[11] * m[13] -
//            m[8]  * m[5] * m[15] +
//            m[8]  * m[7] * m[13] +
//            m[12] * m[5] * m[11] -
//            m[12] * m[7] * m[9];

//   inv[12] = -m[4]  * m[9] * m[14] +
//             m[4]  * m[10] * m[13] +
//             m[8]  * m[5] * m[14] -
//             m[8]  * m[6] * m[13] -
//             m[12] * m[5] * m[10] +
//             m[12] * m[6] * m[9];

//   inv[1] = -m[1]  * m[10] * m[15] +
//            m[1]  * m[11] * m[14] +
//            m[9]  * m[2] * m[15] -
//            m[9]  * m[3] * m[14] -
//            m[13] * m[2] * m[11] +
//            m[13] * m[3] * m[10];

//   inv[5] = m[0]  * m[10] * m[15] -
//            m[0]  * m[11] * m[14] -
//            m[8]  * m[2] * m[15] +
//            m[8]  * m[3] * m[14] +
//            m[12] * m[2] * m[11] -
//            m[12] * m[3] * m[10];

//   inv[9] = -m[0]  * m[9] * m[15] +
//            m[0]  * m[11] * m[13] +
//            m[8]  * m[1] * m[15] -
//            m[8]  * m[3] * m[13] -
//            m[12] * m[1] * m[11] +
//            m[12] * m[3] * m[9];

//   inv[13] = m[0]  * m[9] * m[14] -
//             m[0]  * m[10] * m[13] -
//             m[8]  * m[1] * m[14] +
//             m[8]  * m[2] * m[13] +
//             m[12] * m[1] * m[10] -
//             m[12] * m[2] * m[9];

//   inv[2] = m[1]  * m[6] * m[15] -
//            m[1]  * m[7] * m[14] -
//            m[5]  * m[2] * m[15] +
//            m[5]  * m[3] * m[14] +
//            m[13] * m[2] * m[7] -
//            m[13] * m[3] * m[6];

//   inv[6] = -m[0]  * m[6] * m[15] +
//            m[0]  * m[7] * m[14] +
//            m[4]  * m[2] * m[15] -
//            m[4]  * m[3] * m[14] -
//            m[12] * m[2] * m[7] +
//            m[12] * m[3] * m[6];

//   inv[10] = m[0]  * m[5] * m[15] -
//             m[0]  * m[7] * m[13] -
//             m[4]  * m[1] * m[15] +
//             m[4]  * m[3] * m[13] +
//             m[12] * m[1] * m[7] -
//             m[12] * m[3] * m[5];

//   inv[14] = -m[0]  * m[5] * m[14] +
//             m[0]  * m[6] * m[13] +
//             m[4]  * m[1] * m[14] -
//             m[4]  * m[2] * m[13] -
//             m[12] * m[1] * m[6] +
//             m[12] * m[2] * m[5];

//   inv[3] = -m[1] * m[6] * m[11] +
//            m[1] * m[7] * m[10] +
//            m[5] * m[2] * m[11] -
//            m[5] * m[3] * m[10] -
//            m[9] * m[2] * m[7] +
//            m[9] * m[3] * m[6];

//   inv[7] = m[0] * m[6] * m[11] -
//            m[0] * m[7] * m[10] -
//            m[4] * m[2] * m[11] +
//            m[4] * m[3] * m[10] +
//            m[8] * m[2] * m[7] -
//            m[8] * m[3] * m[6];

//   inv[11] = -m[0] * m[5] * m[11] +
//             m[0] * m[7] * m[9] +
//             m[4] * m[1] * m[11] -
//             m[4] * m[3] * m[9] -
//             m[8] * m[1] * m[7] +
//             m[8] * m[3] * m[5];

//   inv[15] = m[0] * m[5] * m[10] -
//             m[0] * m[6] * m[9] -
//             m[4] * m[1] * m[10] +
//             m[4] * m[2] * m[9] +
//             m[8] * m[1] * m[6] -
//             m[8] * m[2] * m[5];

//   det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

//   if (det == 0)
//     return false;

//   det = 1.0 / det;

//   for (i = 0; i < 16; i++)
//     invOut[i] = inv[i] * det;

//   return true;
// }