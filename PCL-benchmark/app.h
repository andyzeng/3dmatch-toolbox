
struct KeypointMatchEntry
{
	int index;
	string file;
	int x;
	int y;
	vector<float> descriptor;
};

struct DepthImage
{
	Grid2f depths;
	PointCloudf points;
};

struct App
{
	void loadKeypointMatchEntries();
	//void computeAllFPFH();
	//void loadKeypointMatchClouds();
	void computeKeypointDescriptors();
	void computeFinalDescFile();
	void go();

	void computeKeypointDescriptor(KeypointMatchEntry &entry);

	DepthImage makeDepthImage(const string &filename) const;

	set<string> filenameSet;
	//map<string, DepthImage*> allImages;

	vector<KeypointMatchEntry> keypointMatchEntries;
};
