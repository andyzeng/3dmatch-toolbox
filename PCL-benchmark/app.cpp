
#include "main.h"

void App::loadKeypointMatchEntries()
{
	const int maxEntries = 5;

	const string filename = R"(C:\Code\3DMatch\dataset\keyptMtch-10000.7scenes.txt)";
	for (auto &s : util::getFileLines(filename, 3))
	{
		if (s[0] == '#') continue;
		auto parts = util::split(s, '\t');
		KeypointMatchEntry entry;
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
}

DepthImage App::makeDepthImage(const string &filename) const
{
	const string fullFilename = R"(C:\Code\3DMatch\dataset\)" + filename + ".png";
	const float camK[9] = { 585,0,320,0,585,240,0,0,1 };
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

void App::loadKeypointMatchClouds()
{
	for (auto &filename : filenameSet)
	{
		DepthImage *image = new DepthImage();
		*image = makeDepthImage(filename);
		allImages[filename] = image;
	}
}

void App::go()
{
	loadKeypointMatchEntries();
	loadKeypointMatchClouds();
}
