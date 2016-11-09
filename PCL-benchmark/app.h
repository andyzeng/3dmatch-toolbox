
struct App
{
	void go();

	void alignAllSpeakers();
	void testSolver(const SolverParams &params, const FeatureDesc &localFeatureDesc);
	void testSolver(const FeatureDesc &localFeatureDesc);
	void normalizeAllWAVs();
	void createObamaClips();
	void purgeBlacklist();
	void runEvaluations();
	void makeDWAVs();
	void makeTones();

	void makeClusterFiles();
	void makeClusterFile(const string &folderName, const string &outBaseFilename);

	SpeakerDatabase database;
	SpeechAcceleratorEnsemble accelerator;
};
