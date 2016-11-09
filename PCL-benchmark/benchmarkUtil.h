
struct SoundFile;

struct DWAVHelper
{
	static BYTE quantizeDWAV(short v)
	{
		const double scaledValue = math::linearMap(-maxValue, maxValue, -1.0, 1.0, (double)v);
		const double r = math::sign(scaledValue) * log(1.0 + mu * fabs(scaledValue)) / log(1.0 + mu);
		
		return util::boundToByte(math::linearMap(-1.0, 1.0, 0.0, 255.0, r));
	}

	static short unquantizeDWAV(BYTE b)
	{
		const double v = math::linearMap(0.0, 255.0, -1.0, 1.0, (double)b);
		const double s = math::sign(v) * (exp(fabs(v) * log(1.0 + mu)) - 1.0) / mu;
		const double r = math::linearMap(-1.0, 1.0, -maxValue, maxValue, s);
		return util::boundToShort(r);
	}

	static constexpr double mu = 255.0;
	static constexpr double maxValue = 32000.0;
};

class SpeechUtil
{
public:
	static void CSVToWAV(const string &baseFilename)
	{
		CSVToWAV(baseFilename + ".csv", baseFilename + ".wav");
	}
	static void addCSVHeaders(const string &baseFilename);
	static void CSVToWAV(const string &csvFilename, const string &wavOutFilename);
	static vector<short> readWAVFile(const string &filename);

	static void testDWAV(const string &filenameIn, const string &filenameOut);
	static vector<int> convertWAVToDWAV(const string &filenameIn);
	static void convertWAVToDWAV(const string &filenameIn, const string &filenameOut);
	static void convertDWAVToWAV(const string &filenameIn, const string &filenameOut);

	static void saveWAVFile(const string &filename, const vector<short> &values);
	static void saveWAVFile(const string &filename, const vector<short> &values, int start, int end);
	static void normalizeWAVFile(const string &filenameIn, const string &filenameOut);
	static void vizLoudness(const string &filename, const vector<short> &values);

	static Grid2f CSVToGrid(const string &CSVFilename, const float scale = 1.0f);
	static void CSVToBinary(const string &CSVFilename, const string &filenameOut, const float scale = 1.0f);

	static void vectorToCSV(const vector<int> &data, const string &filenameOut);
	static vector<int> CSVToVector(const string &filenameIn);
	
	static void makeSpeechClips(const SoundFile &sound, const string &outDir, const string &outBaseName);

	static void testFLANN(const string &outFilename);

	static ColorImageR8G8B8A8 gridToImage(const Grid2f &image, float low, float high);

	static void makeDeepFeaturePCA();
	static void reduceDeepFeatures(const string &speakerName);

	static EigenSystemf computeEigenSystem(const DenseMatrixf& m);

	static void splitSpeakerParts(const string &sourceDir, const string &speakerName);
};
