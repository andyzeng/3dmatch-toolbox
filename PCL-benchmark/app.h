
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
	static vec3f makePt(int x, int y, float depth);
	void load(const string &filename);
	void filter();
	void saveDebug();
	void makePointCloud();

	Grid2f rawDepths;
	Grid2f smoothDepths;
	Grid2<vec3f> gridPoints;
	PointCloudf cloud;
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

	set<string> filenameSet;
	//map<string, DepthImage*> allImages;

	vector<KeypointMatchEntry> keypointMatchEntries;
};
