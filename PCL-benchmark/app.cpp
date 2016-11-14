
#include "main.h"

//#define USE_FPFH
#define USE_SPIN

const float camK[9] = { 585,0,320,0,585,240,0,0,1 };
const float descriptorRadius = 0.2f;
const float descriptorRadiusSq = descriptorRadius * descriptorRadius;

const int normalK = 50;
const int fpfhK = 200;

const float spinRadius = 0.3f; // fragment
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

const string cacheBase = R"(C:\Code\3DMatch\dataset\cache)" + string("-") + descName + "\\";

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

DepthImage App::makeDepthImage(const string &filename) const
{
	const string fullFilename = R"(C:\Code\3DMatch\dataset\)" + filename + ".depth.png";
	DepthImage16 depthImage;
	FreeImageWrapper::loadImage(fullFilename, depthImage);

	DepthImage result;
	result.depths.allocate(depthImage.getDimensions(), 0.0f);
	for (auto &p : depthImage)
	{
		float depth = float(p.value) / 1000.0f;
		if (depth > 10.0f) // Invalid depth
			depth = 0.0f;

		result.depths(p.x, p.y) = depth;
		if (depth > 0.0f)
		{
			vec3f pt;
			pt.z = depth;
			pt.x = ((float)p.x + 0.5f - camK[0 * 3 + 2]) * depth / camK[0 * 3 + 0];
			pt.y = ((float)p.y + 0.5f - camK[1 * 3 + 2]) * depth / camK[1 * 3 + 1];
			result.points.m_points.push_back(pt);
		}
	}
	return result;
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
	DepthImage image = makeDepthImage(entry.file);

	const float depth = image.depths(entry.x, entry.y);
	vec3f queryPt;
	queryPt.z = depth;
	queryPt.x = ((float)entry.x + 0.5f - camK[0 * 3 + 2]) * depth / camK[0 * 3 + 0];
	queryPt.y = ((float)entry.y + 0.5f - camK[1 * 3 + 2]) * depth / camK[1 * 3 + 1];
	
	PointCloudf localCloud;
	localCloud.m_points.push_back(queryPt);
	for (auto &v : image.points.m_points)
	{
		if (vec3f::distSq(v, queryPt) < descriptorRadiusSq)
		{
			//cout << vec3f::distSq(v, queryPt) << endl;
			localCloud.m_points.push_back(v);
		}
	}

	cout << "Local cloud points: " << localCloud.m_points.size() << endl;
	cout << "Total cloud points: " << image.points.m_points.size() << endl;

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
	util::runSystemCommand("pcl_convert_pcd_ascii_binary_release.exe " + descFilename + " " + asciiFilename + " 0");
	const string outLine = util::getFileLines(asciiFilename, 0)[targetLineIndex];
	auto parts = util::split(outLine, " ");
	parts.resize(descriptorSize);
	ofstream outFile(finalFilename);
	for (auto &f : parts)
		outFile << f << " ";
}

void App::computeKeypointDescriptors()
{
#pragma omp parallel for
	for(int i = 0; i < keypointMatchEntries.size(); i++)
	{
		computeKeypointDescriptor(keypointMatchEntries[i]);
	}
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

	const bool keypointEval = true;
	const bool fragmentEval = false;
	const bool APCEval = false;
	if (keypointEval)
	{
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

#pragma omp parallel for schedule(dynamic)
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
		//fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen1\\");
		//fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen2\\");
		//fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen3\\");
		//fragmentList.push_back(datasetDir + "real\\7-scenes-redkitchen4\\");
		//									 
		//fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c3-hv_c3_1\\");
		//fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c6-hv_c6_1\\");
		//fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c8-hv_c8_3\\");
		fragmentList.push_back(datasetDir + "real\\sun3d-harvard_c11-hv_c11_2\\");
		fragmentList.push_back(datasetDir + "real\\sun3d-hotel_umd-maryland_hotel3\\");

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
