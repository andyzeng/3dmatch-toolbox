
#include "main.h"

const float camK[9] = { 585,0,320,0,585,240,0,0,1 };
const float descriptorRadius = 0.2f;
const float descriptorRadiusSq = descriptorRadius * descriptorRadius;

const int normalK = 50;
const int fpfhK = 200;
const int targetLineIndex = 11;
const int descriptorSize = 33;
const string datasetDir = R"(C:\Code\3DMatch\dataset\)";
const string cacheBase = R"(C:\Code\3DMatch\dataset\cache\)";

const string descOutFilename = "keypoint-FPFH.dat";

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

/*void App::loadKeypointMatchClouds()
{
	for (auto &filename : filenameSet)
	{
		DepthImage *image = new DepthImage();
		*image = makeDepthImage(filename);
		allImages[filename] = image;
		cout << "loaded i" << allImages.size() << endl;
	}
}*/

/*void App::computeAllFPFH()
{
	const int normalK = 50;
	const int fpfhK = 200;
	const string cacheBase = R"(C:\Code\3DMatch\dataset\cache\)";
	for (auto &e : allImages)
	{
		auto &points = e.second->points;
		string fixedFilename = e.first;
		fixedFilename = util::replace(fixedFilename, "data/test/", "");
		fixedFilename = util::replace(fixedFilename, "/", "_");
		const string baseFilename = fixedFilename;
		const string rawFilename = cacheBase + fixedFilename + "-raw.pcd";
		const string normalFilename = cacheBase + fixedFilename + "-n" + to_string(normalK) + ".pcd";
		const string fpfhFilename = cacheBase + fixedFilename + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + ".pcd";
		const string asciiFilename = cacheBase + fixedFilename + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + "-ascii.pcd";

		if (util::fileExists(rawFilename))
		{
			cout << "skipping " << fixedFilename << endl;
			continue;
		}

		cout << "processing " << fixedFilename << endl;
		PointCloudIOf::saveToPCD(rawFilename, points);
		util::runSystemCommand("pcl_normal_estimation_release.exe " + rawFilename + " " + normalFilename + " -k " + to_string(normalK));
		util::runSystemCommand("pcl_fpfh_estimation_release.exe " + normalFilename + " " + fpfhFilename + " -k " + to_string(fpfhK));
		util::runSystemCommand("pcl_convert_pcd_ascii_binary_release.exe " + fpfhFilename + " " + asciiFilename + " 0");
	}
}*/

void App::computeKeypointDescriptor(KeypointMatchEntry &entry)
{
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

	const string entryDesc = "e" + to_string(entry.index);
	const string rawFilename = cacheBase + entryDesc + "-raw.pcd";
	const string normalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + ".pcd";
	const string fpfhFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + ".pcd";
	const string asciiFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + "-ascii.pcd";
	const string finalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + "-desc.txt";

	if (util::fileExists(finalFilename))
	{
		cout << "skipping " << finalFilename << endl;
		return;
	}

	cout << "processing " << entryDesc << endl;
	cout << "query pt: " << queryPt << endl;
	PointCloudIOf::saveToPCD(rawFilename, localCloud);
	util::runSystemCommand("pcl_normal_estimation_release.exe " + rawFilename + " " + normalFilename + " -k " + to_string(normalK));
	util::runSystemCommand("pcl_fpfh_estimation_release.exe " + normalFilename + " " + fpfhFilename + " -k " + to_string(fpfhK));
	util::runSystemCommand("pcl_convert_pcd_ascii_binary_release.exe " + fpfhFilename + " " + asciiFilename + " 0");

	const string outLine = util::getFileLines(asciiFilename, 0)[targetLineIndex];
	auto parts = util::split(outLine, " ");
	parts.resize(descriptorSize);
	ofstream outFile(finalFilename);
	for (auto &f : parts)
		outFile << f << " ";

	//entry.descriptor = PCLUtil::makeFPFHDescriptor(localCloud, queryPt);
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
		const string finalFilename = cacheBase + entryDesc + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + "-desc.txt";
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
	cout << "best dist: " << sqrt(bestDistSq) << endl;
	return bestIndex;
}

void App::processFragment(const string &path)
{
	const string cacheTemp = datasetDir + "cacheFragmentFinal/";
	const int maxPointCount = 50000;
	PointCloudf cloud;
	PointCloudIOf::loadFromPLY(path, cloud);
	std::random_shuffle(cloud.m_points.begin(), cloud.m_points.end());
	if(cloud.m_points.size() > maxPointCount) cloud.m_points.resize(maxPointCount);

	const string filenameOnly = util::removeExtensions(util::getFilenameFromPath(path));
	string fixedPath = util::remove(util::directoryFromPath(path), R"(C:\Code\3DMatch\dataset\synthetic\)");
	fixedPath = util::remove(fixedPath, "/");
	fixedPath = util::remove(fixedPath, "/");
	const string baseName = fixedPath + "_" + filenameOnly;

	const string rawFilename = cacheTemp + baseName + ".pcd";
	const string normalFilename = cacheTemp + baseName + "-n" + to_string(normalK) + ".pcd";
	const string fpfhFilename = cacheTemp + baseName + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + ".pcd";
	const string asciiFilename = cacheTemp + baseName + "-n" + to_string(normalK) + "-fpfh" + to_string(fpfhK) + "-ascii.pcd";
	const string keypointFile = util::replace(path, ".ply", ".keypoints.bin");
	const string descFile = util::replace(path, ".ply", ".keypoints.fpfh.descriptors.bin");

	if (util::fileExists(descFile))
	{
		cout << "skipping " << baseName << endl;
		//return;
	}

	cout << "processing " << baseName << endl;
	PointCloudIOf::saveToPCD(rawFilename, cloud);
	util::runSystemCommand("pcl_normal_estimation_release.exe " + rawFilename + " " + normalFilename + " -k " + to_string(normalK));
	util::runSystemCommand("pcl_fpfh_estimation_release.exe " + normalFilename + " " + fpfhFilename + " -k " + to_string(fpfhK));
	util::runSystemCommand("pcl_convert_pcd_ascii_binary_release.exe " + fpfhFilename + " " + asciiFilename + " 0");

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
	const bool keypointEval = false;
	const bool fragmentEval = true;
	if (keypointEval)
	{
		//loadKeypointMatchEntries();
		//loadKeypointMatchClouds();
		//computeAllFPFH();
		//computeKeypointDescriptors();
		//computeFinalDescFile();
	}
	if (fragmentEval)
	{
		processFragment(R"(C:\Code\3DMatch\dataset\synthetic\iclnuim-livingroom1\cloud_bin_1.ply)");
	}
}
