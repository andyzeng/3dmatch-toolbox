
struct KeypointMatchEntry
{
	string file;
	int x;
	int y;
};

struct DepthImage
{
	Grid2f depths;
	PointCloudf points;
};

struct App
{
	void loadKeypointMatchEntries();
	void loadKeypointMatchClouds();
	void go();

	DepthImage makeDepthImage(const string &filename) const;

	set<string> filenameSet;
	map<string, DepthImage*> allImages;

	vector<KeypointMatchEntry> keypointMatchEntries;
};
