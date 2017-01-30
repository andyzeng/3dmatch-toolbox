
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
	void load(const string &filename);
	void filter();
	void save();

	Grid2f depths;
	Grid2f smoothDepths;
	Grid2<vec3f> gridPoints;
	PointCloudf allPoints;
};

struct App
{
	void loadKeypointMatchEntries();
	//void computeAllFPFH();
	//void loadKeypointMatchClouds();
	void computeKeypointDescriptors();
	void computeFinalDescFile();
	void go();
	void processFragment(const string &path, const string &subfolder);
	void processAllFragments(const string &dir, const string &subfolder);
	void computeBinDescriptor(const string &fileIn, const string &fileOut);
	void computeFinalDescFileTDFs();

	void computeKeypointDescriptor(KeypointMatchEntry &entry);

	DepthImage makeDepthImage(const string &filename) const;

	set<string> filenameSet;
	//map<string, DepthImage*> allImages;

	vector<KeypointMatchEntry> keypointMatchEntries;
};
