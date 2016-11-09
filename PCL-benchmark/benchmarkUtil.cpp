
#include "main.h"

#include "libsndfile/sndfile.h"

#ifdef USE_MATLAB
#include "C:\Program Files\MATLAB\R2014b\extern\include\engine.h"
#pragma comment(lib, "C:\\Program Files\\MATLAB\\R2014b\\extern\\lib\\win64\\microsoft\\libmat.lib")
#pragma comment(lib, "C:\\Program Files\\MATLAB\\R2014b\\extern\\lib\\win64\\microsoft\\libmex.lib")
#pragma comment(lib, "C:\\Program Files\\MATLAB\\R2014b\\extern\\lib\\win64\\microsoft\\libmx.lib")
#pragma comment(lib, "C:\\Program Files\\MATLAB\\R2014b\\extern\\lib\\win64\\microsoft\\libeng.lib")

EigenSystemf SpeechUtil::computeEigenSystem(const DenseMatrixf& m)
{
	Engine* ep;
	if (!(ep = engOpen(NULL))) throw MLIB_EXCEPTION("cannot open matlab engine");

	mxArray* A = mxCreateDoubleMatrix(m.cols(), m.rows(), mxREAL);

	DenseMatrixd dm(m.rows(), m.cols());
	for (size_t i = 0; i < m.rows(); i++) {
		for (size_t j = 0; j < m.cols(); j++) {
			dm(i, j) = m(i, j);
		}
	}

	memcpy((char*)mxGetPr(A), dm.getData(), sizeof(double)*m.cols()*m.rows());

	engPutVariable(ep, "A", A);
	engEvalString(ep, "[V,d] = eigs(A,size(A,1))");
	engEvalString(ep, "D = diag(d)");

	mxArray* D = engGetVariable(ep, "D");
	mxArray* V = engGetVariable(ep, "V");

	double* Dreal = mxGetPr(D);
	double* Vreal = mxGetPr(V);

	engClose(ep);

	EigenSystemf es;
	es.eigenvalues.resize(m.rows());
	for (size_t i = 0; i < m.rows(); i++) {
		es.eigenvalues[i] = (float)Dreal[i];
	}
	es.eigenvectors = DenseMatrixf(m.rows(), m.cols());
	for (size_t i = 0; i < m.rows(); i++) {
		for (size_t j = 0; j < m.cols(); j++) {
			es.eigenvectors(j, i) = (float)Vreal[i*m.cols() + j];
		}
	}

	es.eigenvectors = es.eigenvectors.getTranspose();

	mxDestroyArray(A);
	mxDestroyArray(D);
	mxDestroyArray(V);

	return es;
}

#else

EigenSystemf SpeechUtil::computeEigenSystem(const DenseMatrixf& m)
{
	return m.eigenSystem();
}

#endif

void SpeechUtil::addCSVHeaders(const string &filename)
{
	cout << "Adding headers to " << filename << endl;
	const vector<string> lines = util::getFileLines(filename);
	if (lines[0][0] == 'c') return;

	ofstream file(filename);
	const int colCount = util::split(lines[0], ',').size();
	for (int x = 0; x < colCount; x++)
	{
		file << "c" << util::zeroPad(x, 8);
		if (x != colCount - 1) file << ",";
	}
	file << '\n';
	for (const string &s : lines)
	{
		file << s << '\n';
	}
}

void SpeechUtil::CSVToWAV(const string &csvFilename, const string &wavOutFilename)
{
	if (util::fileExists(wavOutFilename))
	{
		cout << "Skipping " << wavOutFilename << endl;
	}
	auto lines = util::getFileLines(csvFilename);
	vector<short> wav;
	for (auto &line : lines)
	{
		auto parts = util::split(line, ",");
		if (parts[0] == "index") continue;
		if (parts.size() == 3)
			wav.push_back(util::boundToShort(convert::toFloat(parts[2])));
	}
	//cout << wav.size() << endl;
	SpeechUtil::saveWAVFile(wavOutFilename, wav);
}

void SpeechUtil::testDWAV(const string &filenameIn, const string &filenameOut)
{
	ofstream file(filenameOut);
	file << "i,value,quantized,unquantized" << endl;
	const vector<short> vIn = SpeechUtil::readWAVFile(filenameIn);
	vector<BYTE> vOut;
	int i = 0;
	for (short v : vIn)
	{
		BYTE b = DWAVHelper::quantizeDWAV(v);
		short q = DWAVHelper::unquantizeDWAV(b);

		file << i++ << "," << v << "," << (int)b << "," << q << endl;
	}
}

vector<int> SpeechUtil::convertWAVToDWAV(const string &filenameIn)
{
	const vector<short> vIn = SpeechUtil::readWAVFile(filenameIn);
	vector<int> vOut;
	for (short v : vIn)
	{
		BYTE b = DWAVHelper::quantizeDWAV(v);
		vOut.push_back(b);
	}
	return vOut;
}

void SpeechUtil::convertWAVToDWAV(const string &filenameIn, const string &filenameOut)
{
	const vector<short> vIn = SpeechUtil::readWAVFile(filenameIn);
	vector<BYTE> vOut;
	for (short v : vIn)
	{
		BYTE b = DWAVHelper::quantizeDWAV(v);
		vOut.push_back(b);
	}

	util::serializeToFilePrimitive(filenameOut, vOut);
}

void SpeechUtil::convertDWAVToWAV(const string &filenameIn, const string &filenameOut)
{
	vector<BYTE> vIn;
	util::deserializeFromFilePrimitive(filenameIn, vIn);

	vector<short> vOut;
	for (BYTE b : vIn)
	{
		short s = DWAVHelper::unquantizeDWAV(b);
		vOut.push_back(s);
	}

	saveWAVFile(filenameOut, vOut);
}

void SpeechUtil::splitSpeakerParts(const string &sourceDir, const string &speakerName)
{
	const string outDirA = speechParams().databaseDir + speakerName + "A/WAVRaw/";
	const string outDirB = speechParams().databaseDir + speakerName + "B/WAVRaw/";

	util::makeDirectory(outDirA);
	util::makeDirectory(outDirB);

	map< string, vector<string> > baseFileList;
	for (const string &s : Directory::enumerateFiles(sourceDir, ".wav"))
	{
		auto p = util::splitOnFirst(s, "_2016-");
		baseFileList[p.first].push_back(s);
	}
	for (auto &e : baseFileList)
	{
		cout << e.first << " has " << e.second.size() << " files" << endl;
		if (e.second.size() >= 2)
		{
			util::copyFile(sourceDir + e.second[0], outDirA + e.first + ".wav");
			util::copyFile(sourceDir + e.second[1], outDirB + e.first + ".wav");
		}
	}
}

void SpeechUtil::makeDeepFeaturePCA()
{
	const FeatureDesc deepDesc = FeatureDesc::defaultCNN();
	const string pcaFile = speechParams().databaseDir + "PCA-" + deepDesc.deepDesc() + ".dat";

	const int randomFiles = 100;
	const int randomFrames = 100;
	const int totalPoints = randomFiles * randomFrames;
	DenseMatrixf points(totalPoints, deepDesc.deepFeaturesDimensionRaw);

	int pointIndex = 0;

	const vector<string> speakerList = Directory::enumerateDirectories(speechParams().databaseDir);

	if (!util::fileExists(pcaFile))
	{
		cout << "Creating PCA: " << pcaFile << endl;
		for (int fileIndex = 0; fileIndex < randomFiles; fileIndex++)
		{
			const string speakerName = util::randomElement(speakerList);
			const string speakerDir = speechParams().databaseDir + speakerName + "/";
			const string inDir = speakerDir + "deepFeatures-w" + to_string(deepDesc.deepFeaturesWidth) + "-d" + to_string(deepDesc.deepFeaturesDimensionRaw) + "-CSV/";
			const vector<string> fileList = Directory::enumerateFiles(inDir, ".csv");

			const string filename = util::randomElement(fileList);
			const Grid2f g = CSVToGrid(inDir + filename, 1.0f / 1000.0f);
			for (int frameIndex = 0; frameIndex < randomFrames; frameIndex++)
			{
				const int frame = util::randomInteger(50, g.getDimX() - 50);
				for (int d = 0; d < deepDesc.deepFeaturesDimensionRaw; d++)
				{
					points(pointIndex, d) = g(frame, d);
				}
				pointIndex++;
			}
		}

		PCAf pca;
		auto eigenSolver = [](const DenseMatrixf &m) { return SpeechUtil::computeEigenSystem(m); };
		pca.init(points, eigenSolver);
		pca.save(pcaFile);
	}
}

void SpeechUtil::reduceDeepFeatures(const string &speakerName)
{
	const FeatureDesc deepDesc = FeatureDesc::defaultCNN();
	const string pcaFile = speechParams().databaseDir + "PCA-" + deepDesc.deepDesc() + ".dat";

	if (!util::fileExists(pcaFile))
	{
		makeDeepFeaturePCA();
	}

	const string speakerDir = speechParams().databaseDir + speakerName + "/";

	int pointIndex = 0;
	const string inDir = speakerDir + "deepFeatures-w" + to_string(deepDesc.deepFeaturesWidth) + "-d" + to_string(deepDesc.deepFeaturesDimensionRaw) + "-CSV/";
	const string outDir = speakerDir + "deepFeatures-w" + to_string(deepDesc.deepFeaturesWidth) + "-d" + to_string(deepDesc.deepFeaturesDimensionRaw) + "-p" + to_string(deepDesc.deepFeaturesDimensionPCA) + "-DAT/";
	util::makeDirectory(outDir);
	const vector<string> fileList = Directory::enumerateFiles(inDir, ".csv");

	PCAf pca;
	pca.load(pcaFile);

	cout << "Reducing all deep features..." << endl;
	vector<float> descIn(deepDesc.deepFeaturesDimensionRaw), descOut(deepDesc.deepFeaturesDimensionPCA);
	for (const string &filename : fileList)
	{
		const string fileOut = outDir + util::replace(filename, ".csv", ".dat");
		if (util::fileExists(fileOut))
		{
			cout << "Skipping " << fileOut << endl;
		}
		else
		{
			cout << "Creating " << fileOut << endl;
			
			const Grid2f gIn = CSVToGrid(inDir + filename, 1.0f / 1000.0f);
			int frameCount = gIn.getDimX();
			Grid2f gOut(frameCount, deepDesc.deepFeaturesDimensionPCA);

			for (int frame = 0; frame < frameCount; frame++)
			{
				for (int d = 0; d < deepDesc.deepFeaturesDimensionRaw; d++)
				{
					descIn[d] = gIn(frame, d);
				}
				pca.transform(descIn, deepDesc.deepFeaturesDimensionPCA, descOut);
				for (int d = 0; d < deepDesc.deepFeaturesDimensionPCA; d++)
				{
					gOut(frame, d) = descOut[d];
				}
			}
			util::serializeToFilePrimitive(fileOut, gOut);
		}
	}
}

void SpeechUtil::testFLANN(const string &outFilename)
{
	const int descriptorCount = 100;
	const int descriptorDim = 10;
	const int kNearest = 5;
	const int searchCount = 25;
	const int queryIndex = 0;

	float *descriptors = new float[descriptorCount * descriptorDim];
	for (int i = 0; i < descriptorCount; i++)
	{
		for (int j = 0; j < descriptorDim; j++)
		{
			descriptors[i * descriptorDim + j] = util::randomUniformf();
		}
	}
	flann::Matrix<float> dataset(descriptors, descriptorCount, descriptorDim);

	auto getDescriptor = [&](int descriptorIndex) {
		vector<float> result;
		for (int i = 0; i < descriptorDim; i++)
			result.push_back(descriptors[descriptorIndex * descriptorDim + i]);
		return result;
	};

	cout << "Building index..." << endl;
	flann::Index<flann::L2<float> > *FLANNIndex = new flann::Index<flann::L2<float> >(dataset, flann::KDTreeIndexParams(8));
	FLANNIndex->buildIndex();

	flann::Matrix<float> queryStorage;
	flann::Matrix<int> indicesStorage;
	flann::Matrix<float> distsStorage;

	queryStorage = flann::Matrix<float>(new float[descriptorDim], 1, descriptorDim);
	indicesStorage = flann::Matrix<int>(new int[kNearest], 1, kNearest);
	distsStorage = flann::Matrix<float>(new float[kNearest], 1, kNearest);

	memcpy(queryStorage.ptr(), descriptors + queryIndex * descriptorDim, descriptorDim * sizeof(float));

	FLANNIndex->knnSearch(queryStorage, indicesStorage, distsStorage, kNearest, flann::SearchParams(searchCount));

	ofstream file(outFilename);
	file << "index,distReal,distFLANN" << endl;
	for (int i = 0; i < kNearest; i++)
	{
		const float distReal = math::distSqL2(getDescriptor(queryIndex), getDescriptor(indicesStorage[0][i]));
		file << i << "," << distReal << "," << distsStorage[0][i] << endl;
	}
}

void SpeechUtil::vizLoudness(const string &filename, const vector<short> &samples)
{
	ofstream file(filename);
	
	const int sampleCount = samples.size();
	vector<float> loudnessSamples(sampleCount);
	for (auto &x : iterate(loudnessSamples))
	{
		const float loudness = abs((float)samples[x.index] / numeric_limits<short>::max());
		x.value = loudness;
	}

	auto computeAmplitude = [&](int filterSize, int index) {
		float sum = 0.0f;
		for (int x = index - filterSize; x <= index + filterSize; x++)
		{
			if(util::validIndex(loudnessSamples, x))
				sum += loudnessSamples[x];
		}
		sum /= (float)filterSize * 2 + 1;
		return sum;
	};

	vector<int> filterSizes;
	filterSizes.push_back(100);
	filterSizes.push_back(200);
	file << "index,waveform,a100,a200" << endl;
	for (int sample = 0; sample < samples.size(); sample++)
	{
		file << sample << "," << (float)samples[sample] / numeric_limits<short>::max();
		for (int f : filterSizes)
		{
			file << "," << computeAmplitude(f, sample);
		}
		file << endl;
	}
}

void SpeechUtil::normalizeWAVFile(const string &filenameIn, const string &filenameOut)
{
	if (util::fileExists(filenameOut))
	{
		cout << "Skipping " << filenameOut << endl;
		return;
	}
	vector<short> waveform = readWAVFile(filenameIn);
	double sum = 0.0;
	for (short v : waveform) sum += abs(v);
	sum /= waveform.size();
	cout << "avg amplitude: " << filenameIn << " " << sum << endl;
	
	const double scale = Constants::normalizedAmplitude / sum;
	for (short &vS : waveform)
	{
		int v = math::round((double)vS * scale);
		if (v <= numeric_limits<short>::min()) vS = numeric_limits<short>::min();
		else if (v >= numeric_limits<short>::max()) vS = numeric_limits<short>::max();
		else vS = (short)v;
	}

	saveWAVFile(filenameOut, waveform);
}

vector<short> SpeechUtil::readWAVFile(const string &filename)
{
	SF_INFO info;
	SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &info);

	if (file == nullptr)
	{
		cout << "Failed to open " << filename << endl;
		return vector<short>();
	}

	vector<short> data(info.frames);
	sf_read_short(file, data.data(), info.frames);
	sf_close(file);

	return data;
}

void SpeechUtil::saveWAVFile(const string &filename, const vector<short> &values)
{
	SF_INFO infoOut;
	infoOut.channels = 1;
	infoOut.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	infoOut.samplerate = 16000;
	SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &infoOut);
	sf_write_short(file, values.data(), values.size());
	sf_close(file);
}

void SpeechUtil::saveWAVFile(const string &filename, const vector<short> &values, int start, int end)
{
	if (start < 0) start = 0;
	if (end >= values.size()) end = (int)values.size() - 1;
	SF_INFO infoOut;
	infoOut.channels = 1;
	infoOut.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	infoOut.samplerate = 16000;
	SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &infoOut);
	sf_write_short(file, values.data() + start, end - start);
	sf_close(file);
}

vector<int> SpeechUtil::CSVToVector(const string &filenameIn)
{
	auto line = util::getFileLines(filenameIn)[1];
	vector<int> result;
	for (auto &s : util::split(line, ','))
	{
		result.push_back(convert::toInt(s));
	}
	return result;
}

void SpeechUtil::vectorToCSV(const vector<int> &data, const string &filenameOut)
{
	ofstream file(filenameOut);
	const int dimX = data.size();
	for (int x = 0; x < dimX; x++)
	{
		file << "c" << util::zeroPad(x, 10);
		if (x != dimX - 1) file << ",";
	}
	file << endl;
	for (int x = 0; x < dimX; x++)
	{
		file << data[x];
		if (x != dimX - 1) file << ",";
	}
	file << endl;
}

Grid2f SpeechUtil::CSVToGrid(const string &CSVFilename, const float scale)
{
	cout << "loading CSV: " << CSVFilename << endl;

	const auto lines = util::getFileLines(CSVFilename, 3);
	const int xDim = (int)util::split(lines[0], ',').size();
	const int yDim = (int)lines.size();

	Grid2f g(xDim, yDim);
	for (int y = 0; y < yDim; y++)
	{
		auto parts = util::split(lines[y], ',');
		for (int x = 0; x < xDim; x++)
		{
			g(x, y) = convert::toFloat(parts[x]) * scale;
		}
	}
	return g;
}

void SpeechUtil::CSVToBinary(const string &CSVFilename, const string &filenameOut, float scale)
{
	if (util::fileExists(filenameOut)) return;

	cout << "converting CSV: " << CSVFilename << endl;

	const auto lines = util::getFileLines(CSVFilename, 3);
	const int xDim = (int)util::split(lines[0], ',').size();
	const int yDim = (int)lines.size();
	
	Grid2f g(xDim, yDim);
	for (int y = 0; y < yDim; y++)
	{
		auto parts = util::split(lines[y], ',');
		for (int x = 0; x < xDim; x++)
		{
			g(x, y) = convert::toFloat(parts[x]) * scale;
		}
	}
	util::serializeToFilePrimitive(filenameOut, g);
}

ColorImageR8G8B8A8 SpeechUtil::gridToImage(const Grid2f &g, float low, float high)
{
	const ColorGradient gradient(LodePNG::load("C:/code/mLib/data/colormaps/parula.png"));

	ColorImageR8G8B8A8 result((int)g.getDimX(), (int)g.getDimY());
	for (auto &p : result)
	{
		const float v = g(p.x, p.y);
		const RGBColor color = gradient.value(math::linearMap(low, high, 0.0f, 1.0f, v));
		p.value = color;
	}
	return result;
}

void SpeechUtil::makeSpeechClips(const SoundFile &sound, const string &outDir, const string &outBaseName)
{
	const float soundThreshold = 0.56f;
	const float silenceDurationThreshold = 0.4f;
	const int smoothingFrameRadius = 6;
	const float minClipSeconds = 2.0f;
	const float maxClipSeconds = 6.0f;
	const int bufferFrames = 25;

	enum class Annotation
	{
		Speech,
		Silence,
	};
	struct Range
	{
		Annotation annotation;
		int frameStart;
		int frameEnd; // exclusive

		double frameDuration() { return (frameEnd - frameStart) * 0.01; }
	};

	util::makeDirectory(outDir);

	const int frameCount = sound.frameCount;

	vector<Annotation> frameAnnotationRaw(frameCount);
	for (int f = 0; f < frameCount; f++)
	{
		const float avgValue = sound.frameAvgValue(f, SoundFeatureType::Joint);
		if (avgValue <= soundThreshold) frameAnnotationRaw[f] = Annotation::Silence;
		else frameAnnotationRaw[f] = Annotation::Speech;
	}

	vector<Annotation> frameAnnotationFiltered(frameCount);
	for (int f = 0; f < frameCount; f++)
	{
		float sum = 0.0f;
		for (int offset = -smoothingFrameRadius; offset <= smoothingFrameRadius; offset++)
		{
			Annotation annotation = util::clampedRead(frameAnnotationRaw, f + offset);
			if (annotation == Annotation::Speech) sum += 1.0f;
			else sum -= 1.0f;
		}
		sum /= (smoothingFrameRadius * 2 + 1);
		if (sum >= 0.0f) frameAnnotationFiltered[f] = Annotation::Speech;
		else frameAnnotationFiltered[f] = Annotation::Silence;
	}

	Range activeRange;
	activeRange.frameStart = 0;
	activeRange.frameEnd = 1;
	activeRange.annotation = Annotation::Silence;
	vector<Range> ranges;
	for (int f = 0; f < frameCount; f++)
	{
		if (frameAnnotationFiltered[f] == activeRange.annotation)
		{
			activeRange.frameEnd = f + 1;
		}
		else
		{
			ranges.push_back(activeRange);
			activeRange.frameStart = f;
			activeRange.frameEnd = f + 1;
			activeRange.annotation = frameAnnotationFiltered[f];
		}
	}
	ranges.push_back(activeRange);

	for (int r = 1; r < ranges.size() - 1; r++)
	{
		Range prev = ranges[r - 1];
		Range current = ranges[r];
		Range next = ranges[r + 1];

		if (prev.annotation == Annotation::Silence && prev.frameDuration() >= silenceDurationThreshold &&
			next.annotation == Annotation::Silence && next.frameDuration() >= silenceDurationThreshold &&
			current.annotation == Annotation::Speech && current.frameDuration() >= minClipSeconds && current.frameDuration() <= maxClipSeconds)
		{
			const string desc = to_string(current.frameStart) + "_" + to_string(current.frameEnd);
			const string filename = outDir + outBaseName + "_" + desc + ".wav";

			if (util::fileExists(filename))
			{
				cout << "Skipped " << filename << endl;
			}
			else
			{
				vector<short> waveform;
				for (int frame = current.frameStart - bufferFrames; frame < current.frameEnd + bufferFrames; frame++)
				{
					for (int sample = 0; sample < Constants::samplesPerFrame; sample++)
					{
						const short value = sound.sampleWaveformShort(frame * Constants::samplesPerFrame + sample);
						waveform.push_back(value);
					}
				}
				SpeechUtil::saveWAVFile(filename, waveform);
				cout << "Saved " << filename << endl;
			}
		}
	}
}
