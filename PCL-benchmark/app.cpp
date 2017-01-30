
#include "main.h"

//#define USE_FPFH
//#define USE_SPIN
//#define USE_SHOT
//#define USE_VFH
//#define USE_CVFH
#define USE_RSD

const float camK[9] = { 585,0,320,0,585,240,0,0,1 };
const float descriptorRadius = 0.4f;
const float descriptorRadiusSq = descriptorRadius * descriptorRadius;

const int normalK = 50;
const int fpfhK = 1500;

const float spinRadius = 20.0f; // keypoint-tsdf
//const float spinRadius = 0.3f; // fragment
//const float spinRadius = 0.05f; // APC

const int targetLineIndex = 11;
const string datasetDir = R"(C:\Code\3DMatch\dataset\)";

#ifdef USE_FPFH
const bool useFPFH = true;
const bool useSpin = false;
const string descName = "fpfh";
const string descOutFilename = "keypoint-FPFH.dat";
const int descriptorSize = 33;
#endif

#ifdef USE_SPIN
const bool useFPFH = false;
const bool useSpin = true;
const string descName = "spin";
const string descOutFilename = "keypoint-spin.dat";
const int descriptorSize = 153;
#endif

#ifdef USE_SHOT
const bool useFPFH = false;
const bool useSpin = false;
const bool useShot = true;
const bool useVFH = false;
const string descName = "shot";
const string descOutFilename = "keypoint-SHOT.dat";
const int descriptorSize = 352;
#endif

#ifdef USE_VFH
const bool useFPFH = false;
const bool useSpin = false;
const bool useShot = false;
const bool useVFH = true;
const bool useCVFH = false;
const string descName = "vfh";
const string descOutFilename = "keypoint-vfh-k100.dat";
const int descriptorSize = 308;
#endif

#ifdef USE_CVFH
const bool useFPFH = false;
const bool useSpin = false;
const bool useShot = false;
const bool useVFH = false;
const bool useCVFH = true;
const string descName = "cvfh";
const string descOutFilename = "keypoint-cvfh.dat";
const int descriptorSize = 308;
#endif

#ifdef USE_RSD
const bool useFPFH = false;
const bool useSpin = false;
const bool useShot = false;
const bool useVFH = false;
const bool useCVFH = false;
const bool useRSD = true;
const string descName = "rsd";
const string descOutFilename = "keypoint-rsd-060.dat";
const int descriptorSize = 2;
#endif

const string cacheBase = R"(D:\Code\3DMatch\dataset\cacheRSD-060)" + string("-") + descName + "\\";

const string problemName = "APC";

void App::loadKeypointMatchEntries()
{
	const int maxEntries = 5000000;

	const string filename = R"(C:\Code\3DMatch\dataset\keyptMtch-10000.7scenes.txt)";
	for (auto &s : util::getFileLines(filename, 3))
	{
		if (s[0] == '#') continue;
		auto parts = util::split(s, '\t');
		KeypointMatchEntry entry;
		entry.index = convert::toInt(parts[0]);
		entry.file = parts[1];
		entry.x = convert::toInt(parts[2]);
		entry.y = convert::toInt(parts[3]);
		keypointMatchEntries.push_back(entry);

		filenameSet.insert(entry.file);

		if (keypointMatchEntries.size() > maxEntries)
		{
			break;
		}
	}

	cout << filenameSet.size() << " unique depth files" << endl;
}


float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist) / (2.0*sigma*sigma));
}

float linearR(float sigma, float dist)
{
	return max(1.0f, min(0.0f, 1.0f - (dist*dist) / (2.0f*sigma*sigma)));
}

float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
}

float gaussD(float sigma, int x)
{
	return exp(-((x*x) / (2.0f*sigma*sigma)));
}

vec3f DepthImage::makePt(int x, int y, float depth)
{
	vec3f pt;
	pt.z = depth;
	pt.x = ((float)x + 0.5f - camK[0 * 3 + 2]) * depth / camK[0 * 3 + 0];
	pt.y = ((float)y + 0.5f - camK[1 * 3 + 2]) * depth / camK[1 * 3 + 1];
	return pt;
}

void DepthImage::filter()
{
	const float inf = numeric_limits<float>::infinity();
	const float depthSigmaD = 2.0f;
	const float depthSigmaR = 0.1f;
	const int kernelRadius = (int)ceil(2.0*depthSigmaD);

	smoothDepths = rawDepths;
	smoothDepths.setValues(inf);

	const int dimX = smoothDepths.getDimX();
	const int dimY = smoothDepths.getDimY();
	

	for (auto &pSmooth : smoothDepths)
	{
		int xCenter = pSmooth.x;
		int yCenter = pSmooth.y;
		
		float sum = 0.0f;
		float sumWeight = 0.0f;

		const float depthCenter = rawDepths(xCenter, yCenter);
		if (depthCenter > 0.0f && depthCenter != inf)
		{
			for (int m = xCenter - kernelRadius; m <= xCenter + kernelRadius; m++)
			{
				for (int n = yCenter - kernelRadius; n <= yCenter + kernelRadius; n++)
				{
					if (m >= 0 && n >= 0 && m < dimX && n < dimY)
					{
						const float currentDepth = rawDepths(xCenter, yCenter);

						if (currentDepth > 0.0f && currentDepth != inf)
						{
							const float weight = gaussD(depthSigmaD, m - xCenter, n - yCenter) * gaussR(depthSigmaR, currentDepth - depthCenter);

							sumWeight += weight;
							sum += weight*currentDepth;
						}
					}
				}
			}

			if (sumWeight > 0.0f) pSmooth.value = sum / sumWeight;
		}
	}
}

void DepthImage::saveDebug()
{
	PointCloudIOf::saveToPLY(R"(C:\Code\3DMatch\dataset\debug.ply)", cloud);
}

void DepthImage::makePointCloud()
{
	for (auto &p : smoothDepths)
	{
		float depth = p.value;
		if (depth > 0.0f && depth != numeric_limits<float>::infinity())
		{
			const vec3f pt = makePt(p.x, p.y, depth);
			cloud.m_points.push_back(pt);
		}
	}
}

void DepthImage::load(const string &filename)
{
	const string fullFilename = R"(C:\Code\3DMatch\dataset\)" + filename + ".depth.png";
	DepthImage16 depthImage;
	FreeImageWrapper::loadImage(fullFilename, depthImage);

	rawDepths.allocate(depthImage.getDimensions(), 0.0f);
	for (auto &p : depthImage)
	{
		float depth = float(p.value) / 1000.0f;
		if (depth > 10.0f) // Invalid depth
			depth = 0.0f;

		rawDepths(p.x, p.y) = depth;
		/*if (depth > 0.0f)
		{
			vec3f pt;
			pt.z = depth;
			pt.x = ((float)p.x + 0.5f - camK[0 * 3 + 2]) * depth / camK[0 * 3 + 0];
			pt.y = ((float)p.y + 0.5f - camK[1 * 3 + 2]) * depth / camK[1 * 3 + 1];
			result.points.m_points.push_back(pt);
		}*/
	}

	filter();
	makePointCloud();
}

void App::computeKeypointDescriptor(KeypointMatchEntry &entry)
{
	const string entryDesc = "e" + to_string(entry.index);
	const string rawFilename = cacheBase + entryDesc + "-raw.pcd";
	const string normalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + ".pcd";
	const string descFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + ".pcd";
	const string asciiFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + "-ascii.pcd";
	const string finalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + "-desc.txt";

	if (util::fileExists(finalFilename))
	{
		cout << "skipping " << finalFilename << endl;
		return;
	}

	//DepthImage &image = *allImages[entry.file];
	DepthImage image;
	image.load(entry.file);
	
	const float depth = image.rawDepths(entry.x, entry.y);
	vec3f queryPt;
	queryPt.z = depth;
	queryPt.x = ((float)entry.x + 0.5f - camK[0 * 3 + 2]) * depth / camK[0 * 3 + 0];
	queryPt.y = ((float)entry.y + 0.5f - camK[1 * 3 + 2]) * depth / camK[1 * 3 + 1];
	
	PointCloudf localCloud;
	localCloud.m_points.push_back(queryPt);
	for (auto &v : image.cloud.m_points)
	{
		if (vec3f::distSq(v, queryPt) < descriptorRadiusSq)
		{
			//cout << vec3f::distSq(v, queryPt) << endl;
			localCloud.m_points.push_back(v);
		}
	}

	cout << "Local cloud points: " << localCloud.m_points.size() << endl;
	cout << "Total cloud points: " << image.cloud.m_points.size() << endl;

	cout << "processing " << entryDesc << endl;
	cout << "query pt: " << queryPt << endl;
	PointCloudIOf::saveToPCD(rawFilename, localCloud);
	util::runSystemCommand("pcl_normal_estimation_release.exe " + rawFilename + " " + normalFilename + " -k " + to_string(normalK));
	if (useFPFH)
	{
		util::runSystemCommand("pcl_fpfh_estimation_release.exe " + normalFilename + " " + descFilename + " -k " + to_string(fpfhK));
	}
	if (useSpin)
	{
		util::runSystemCommand("pcl_spin_estimation_release.exe " + normalFilename + " " + descFilename + " -radius " + to_string(spinRadius));
	}
	if (useShot)
	{
		util::runSystemCommand("pcl_shot_estimation_release.exe " + normalFilename + " " + descFilename + " -radius " + to_string(spinRadius));
	}
	if (useVFH)
	{
		util::runSystemCommand("pcl_vfh_estimation_release.exe " + normalFilename + " " + descFilename);
	}
	if (useCVFH)
	{
		util::runSystemCommand("pcl_cvfh_estimation_release.exe " + normalFilename + " " + descFilename);
	}
	if (useRSD)
	{
		util::runSystemCommand("pcl_rsd_estimation_release.exe " + normalFilename + " " + descFilename);
	}
	util::runSystemCommand("pcl_convert_pcd_ascii_binary_release.exe " + descFilename + " " + asciiFilename + " 0");

	ofstream outFile(finalFilename);
	const string outLine = util::getFileLines(asciiFilename, 0)[targetLineIndex];
	auto parts = util::split(outLine, " ");
	parts.resize(descriptorSize);
	for (auto &f : parts)
		outFile << f << " ";
}

void App::computeKeypointDescriptors()
{
#pragma omp parallel for schedule(dynamic) num_threads(32)
	for(int i = 0; i < keypointMatchEntries.size(); i++)
	{
		computeKeypointDescriptor(keypointMatchEntries[i]);
	}
}

void App::computeFinalDescFileTDFs()
{
	const int descCount = 15000;
	vector< vector<float> > descriptors(descCount);
	for (int i = 0; i < descCount; i++)
	{
		//0-n50-spin1000-desc.txt
		const string descFilename = cacheBase + to_string(i) + "-n100-" + descName + "1500-desc.txt";
		const string line = util::getFileLines(descFilename)[0];
		auto parts = util::split(line, " ");
		for (int j = 0; j < descriptorSize; j++)
			descriptors[i].push_back(convert::toFloat(parts[j]));

		if (i % 100 == 0)
		{
			cout << "Loaded " << i << endl;
		}
	}

	FILE *fOut = util::checkedFOpen(datasetDir + "TDF-" + descOutFilename, "wb");
	float descCountF = descCount;
	float descriptorDim = descriptorSize;
	util::checkedFWrite(&descCountF, 4, 1, fOut);
	util::checkedFWrite(&descriptorDim, 4, 1, fOut);
	for (int i = 0; i < descCount; i++)
	{
		util::checkedFWrite(descriptors[i].data(), 4, descriptorDim, fOut);
	}
	fclose(fOut);
}

void App::computeFinalDescFile()
{
	for (auto &e : keypointMatchEntries)
	{
		const string entryDesc = "e" + to_string(e.index);
		const string finalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + "-desc.txt";
		const string line = util::getFileLines(finalFilename)[0];
		auto parts = util::split(line, " ");
		for (int i = 0; i < descriptorSize; i++)
			e.descriptor.push_back(convert::toFloat(parts[i]));

		if (e.index % 100 == 0)
		{
			cout << "Loaded " << e.index << endl;
		}
	}

	FILE *fOut = util::checkedFOpen(datasetDir + descOutFilename, "wb");
	float pointCount = keypointMatchEntries.size();
	float descriptorDim = descriptorSize;
	util::checkedFWrite(&pointCount, 4, 1, fOut);
	util::checkedFWrite(&descriptorDim, 4, 1, fOut);
	for (auto &e : keypointMatchEntries)
	{
		util::checkedFWrite(e.descriptor.data(), 4, descriptorDim, fOut);
	}
	fclose(fOut);
}

int findClosestIndex(const PointCloudf &cloud, const vec3f &v)
{
	int bestIndex = -1;
	float bestDistSq = numeric_limits<float>::max();
	for (int i = 0; i < cloud.m_points.size(); i++)
	{
		const float d = vec3f::distSq(cloud.m_points[i], v);
		if (d < bestDistSq)
		{
			bestDistSq = d;
			bestIndex = i;
		}
	}
	//cout << "best dist: " << sqrt(bestDistSq) << endl;
	return bestIndex;
}

void App::processAllFragments(const string &dir, const string &subfolder)
{
	vector<string> allFiles = Directory::enumerateFilesWithPath(dir, ".ply");
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < allFiles.size(); i++)
	{
		processFragment(allFiles[i], subfolder);
	}
}

PointCloudf makePointCloudFromBin(const string &filenameIn)
{
	FILE *fileIn = util::checkedFOpen(filenameIn, "rb");
	float n;
	util::checkedFRead(&n, 4, 1, fileIn);

	PointCloudf result;
	result.m_points.push_back(vec3f::origin);
	cout << "loading " << n << " keypoints" << endl;
	for (int i = 0; i < n; i++)
	{
		vec3f v;
		util::checkedFRead(&v, sizeof(float), 3, fileIn);
		result.m_points.push_back(v);
	}
	fclose(fileIn);
	return result;
}

void App::computeBinDescriptor(const string &filenameIn, const string &filenameOut)
{
	const string entryDesc = util::removeExtensions(util::fileNameFromPath(filenameIn));
	const string rawFilename = cacheBase + entryDesc + "-raw.pcd";
	const string normalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + ".pcd";
	const string descFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + ".pcd";
	const string asciiFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + "-ascii.pcd";
	const string finalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + "-desc.txt";

	if (util::fileExists(finalFilename))
	{
		cout << "skipping " << finalFilename << endl;
		return;
	}

	const PointCloudf cloud = makePointCloudFromBin(filenameIn);
	cout << "Local cloud points: " << cloud.m_points.size() << endl;
	//PointCloudIOf::saveToPLY("test.ply", cloud);
	//return;
	
	cout << "processing " << entryDesc << endl;
	PointCloudIOf::saveToPCD(rawFilename, cloud);
	util::runSystemCommand("pcl_normal_estimation_release.exe " + rawFilename + " " + normalFilename + " -k " + to_string(normalK));
	if (useFPFH)
	{
		util::runSystemCommand("pcl_fpfh_estimation_release.exe " + normalFilename + " " + descFilename + " -k " + to_string(fpfhK));
	}
	if (useSpin)
	{
		util::runSystemCommand("pcl_spin_estimation_release.exe " + normalFilename + " " + descFilename + " -radius " + to_string(spinRadius));
	}
	if (useShot)
	{
		util::runSystemCommand("pcl_shot_estimation_release.exe " + normalFilename + " " + descFilename + " -radius " + to_string(spinRadius));
	}
	if (useVFH)
	{
		util::runSystemCommand("pcl_vfh_estimation_release.exe " + normalFilename + " " + descFilename);
	}
	if (useCVFH)
	{
		util::runSystemCommand("pcl_cvfh_estimation_release.exe " + normalFilename + " " + descFilename);
	}
	if (useRSD)
	{
		util::runSystemCommand("pcl_rsd_estimation_release.exe " + normalFilename + " " + descFilename);
	}
	util::runSystemCommand("pcl_convert_pcd_ascii_binary_release.exe " + descFilename + " " + asciiFilename + " 0");
	const string outLine = util::getFileLines(asciiFilename, 0)[targetLineIndex];
	auto parts = util::split(outLine, " ");
	parts.resize(descriptorSize);
	ofstream outFile(finalFilename);
	for (auto &f : parts)
		outFile << f << " ";

}

void App::processFragment(const string &path, const string &subfolder)
{
	const string cacheTemp = datasetDir + "cacheFragmentTemp-" + problemName + "-" + descName + "/";
	const string descOutDir = datasetDir + "fragmentOut-" + problemName + "-" + descName + "/" + subfolder + "/";
	util::makeDirectory(cacheTemp);
	util::makeDirectory(descOutDir);
	const int maxPointCount = 30000;
	
	const string filenameOnly = util::removeExtensions(util::getFilenameFromPath(path));
	string fixedPath = util::remove(util::directoryFromPath(path), R"(C:\Code\3DMatch\dataset\)");
	fixedPath = util::replace(fixedPath, "/", "_");
	fixedPath = util::replace(fixedPath, "\\", "_");
	const string baseName = fixedPath + filenameOnly;

	const string rawFilename = cacheTemp + baseName + ".pcd";
	const string normalFilename = cacheTemp + baseName + "-n" + to_string(normalK) + ".pcd";
	const string descFilename = cacheTemp + baseName + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + ".pcd";
	const string asciiFilename = cacheTemp + baseName + "-n" + to_string(normalK) + "-" + descName + to_string(fpfhK) + "-ascii.pcd";
	const string keypointFile = util::replace(path, ".ply", ".keypoints.bin");
	const string descFile = descOutDir + filenameOnly + ".keypoints." + descName + ".descriptors.bin";

	if (util::fileExists(descFile))
	{
		cout << "skipping " << baseName << endl;
		return;
	}

	PointCloudf cloud;
	PointCloudIOf::loadFromPLY(path, cloud);
	//PointCloudIOf::saveToPLY(path + "echo", cloud);
	std::random_shuffle(cloud.m_points.begin(), cloud.m_points.end());
	if (cloud.m_points.size() > maxPointCount) cloud.m_points.resize(maxPointCount);

	cout << "processing " << baseName << endl;
	PointCloudIOf::saveToPCD(rawFilename, cloud);
	util::runSystemCommand("pcl_normal_estimation_release.exe " + rawFilename + " " + normalFilename + " -k " + to_string(normalK));
	if (useFPFH)
	{
		util::runSystemCommand("pcl_fpfh_estimation_release.exe " + normalFilename + " " + descFilename + " -k " + to_string(fpfhK));
	}
	if (useSpin)
	{
		util::runSystemCommand("pcl_spin_estimation_release.exe " + normalFilename + " " + descFilename + " -radius " + to_string(spinRadius));
	}
	util::runSystemCommand("pcl_convert_pcd_ascii_binary_release.exe " + descFilename + " " + asciiFilename + " 0");

	auto allDescLines = util::getFileLines(asciiFilename, 0);
	
	FILE *fileIn = util::checkedFOpen(keypointFile, "rb");
	FILE *fileOut = util::checkedFOpen(descFile, "wb");
	float n;
	util::checkedFRead(&n, 4, 1, fileIn);

	float x = n, y = descriptorSize;
	util::checkedFWrite(&x, sizeof(float), 1, fileOut);
	util::checkedFWrite(&y, sizeof(float), 1, fileOut);


	cout << "loading " << n << " keypoints" << endl;
	for (int i = 0; i < n; i++)
	{
		vec3f v;
		util::checkedFRead(&v, sizeof(float), 3, fileIn);
		int closestIdx = findClosestIndex(cloud, v);
		const string descLine = allDescLines[targetLineIndex + closestIdx];
		auto parts = util::split(descLine, " ");
		parts.resize(descriptorSize);
		for (auto &f : parts)
		{
			float v = convert::toFloat(f);
			util::checkedFWrite(&v, sizeof(float), 1, fileOut);
		}
	}

	fclose(fileIn);
	fclose(fileOut);
}

void App::go()
{
	util::makeDirectory(cacheBase);

	const bool testDepthMap = true;
	const bool keypointEval = false;
	const bool fragmentEval = false;
	const bool APCEval = false;
	if (testDepthMap)
	{
		DepthImage image;
		image.load(R"(data\test\7-scenes-redkitchen\seq-01\frame-000006)");
		image.saveDebug();
	}
	if (keypointEval)
	{
		/*vector<string> fragmentList;
		for (auto &s : Directory::enumerateFilesWithPath(R"(C:\Code\3DMatch\dataset\for-matt)", ".bin"))
		{
			fragmentList.push_back(s);
		}

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < fragmentList.size(); i++)
		{
			//computeBinDescriptor(fragmentList[i], "");
		}*/

		//computeFinalDescFileTDFs();
		loadKeypointMatchEntries();
		computeKeypointDescriptors();
		computeFinalDescFile();
	}
	if (APCEval)
	{
		processAllFragments(R"(C:\Code\3DMatch\dataset\apc\objects\)", "objects");

		vector<string> fragmentList;
		for (auto &s : Directory::enumerateDirectoriesWithPath(R"(C:\Code\3DMatch\dataset\apc\scenarios)"))
		{
			fragmentList.push_back(s);
		}
		//num_threads(7)
#pragma omp parallel for schedule(dynamic) num_threads(16)
		for (int i = 0; i < fragmentList.size(); i++)
		{
			const string fragmentName = util::split(fragmentList[i], "\\").back();
			processAllFragments(fragmentList[i], fragmentName);
		}
	}
	if (fragmentEval)
	{
		vector<string> fragmentList;
		//fragmentList.push_back(datasetDir + "synthetic\\iclnuim-livingroom1\\");
		//fragmentList.push_back(datasetDir + "synthetic\\iclnuim-livingroom2\\");
		//fragmentList.push_back(datasetDir + "synthetic\\iclnuim-office1\\");
		//fragmentList.push_back(datasetDir + "synthetic\\iclnuim-office2\\");
		//
		fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen1\\");
		//fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen2\\");
		//fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen3\\");
		//fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen4\\");
		//									 
		//fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c3-hv_c3_1\\");
		//fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c6-hv_c6_1\\");
		//fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c8-hv_c8_3\\");
		//fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c11-hv_c11_2\\");
		//fragmentList.push_back(datasetDir + "real\\sun3d-hotel_umd-maryland_hotel3\\");

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < fragmentList.size(); i++)
		{
			const string fragmentName = util::split(fragmentList[i], "\\").back();
			processAllFragments(fragmentList[i], fragmentName);
		}

		//processFragment(R"(C:\Code\3DMatch\dataset\synthetic\iclnuim-livingroom1\cloud_bin_1.ply)");
		//return;
		/*processAllFragments(datasetDir + "synthetic\\iclnuim-livingroom1\\", "iclnuim-livingroom1");
		processAllFragments(datasetDir + "synthetic\\iclnuim-livingroom2\\", "iclnuim-livingroom2");
		processAllFragments(datasetDir + "synthetic\\iclnuim-office1\\", "iclnuim-office1");
		processAllFragments(datasetDir + "synthetic\\iclnuim-office2\\", "iclnuim-office2");

		processAllFragments(datasetDir + "real\\7-scenes-redkitchen1\\", "7-scenes-redkitchen1");
		processAllFragments(datasetDir + "real\\7-scenes-redkitchen2\\", "7-scenes-redkitchen2");
		processAllFragments(datasetDir + "real\\7-scenes-redkitchen3\\", "7-scenes-redkitchen3");
		processAllFragments(datasetDir + "real\\7-scenes-redkitchen4\\", "7-scenes-redkitchen4");

		processAllFragments(datasetDir + "real\\sun3d-harvard_c3-hv_c3_1\\", "sun3d-harvard_c3-hv_c3_1");
		processAllFragments(datasetDir + "real\\sun3d-harvard_c6-hv_c6_1\\", "sun3d-harvard_c6-hv_c6_1");
		processAllFragments(datasetDir + "real\\sun3d-harvard_c11-hv_c11_2\\", "sun3d-harvard_c11-hv_c11_2");
		processAllFragments(datasetDir + "real\\sun3d-harvard_c11-hv_c11_2\\", "sun3d-harvard_c11-hv_c11_2");
		processAllFragments(datasetDir + "real\\sun3d-hotel_umd-maryland_hotel3\\", "sun3d-hotel_umd-maryland_hotel3");*/
	}
}
