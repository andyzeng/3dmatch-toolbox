
#include "main.h"

void App::normalizeAllWAVs()
{
	for (const string &speakerDir : Directory::enumerateDirectoriesWithPath(speechParams().databaseDir))
	{
		util::makeDirectory(speakerDir + "WAVNorm/");
		util::makeDirectory(speakerDir + "featuresCSV/");
		for (const string &WAVFilename : Directory::enumerateFiles(speakerDir + "WAVRaw"))
		{
			SpeechUtil::normalizeWAVFile(speakerDir + "WAVRaw/" + WAVFilename, speakerDir + "WAVNorm/" + WAVFilename);
		}
	}
}

void App::testSolver(const SolverParams &params, const FeatureDesc &localFeatureDesc)
{
	SpeakerEntry &querySpeaker = *database.speakers[speechParams().querySpeakerName];
	SpeakerEntry &targetSpeaker = *database.speakers[speechParams().targetSpeakerName];

	const int testFileIndex = database.getFileIndex(speechParams().queryFilename);
	//const int testFileIndex = database.getFileIndex("arctic_a0005");
	
	const SoundFile queryFile = querySpeaker.getFile(testFileIndex).clone();
	
	const string queryFilename = speechParams().debugDir + "query";
	queryFile.saveWAV(queryFilename);
	queryFile.saveFeatureImage(queryFilename, true);

	WaveformSolver solver;
	solver.solve(database, queryFile, testFileIndex, targetSpeaker, accelerator, localFeatureDesc, params);
	//solver.dumpCandidates(querySpeaker);
	solver.dumpSolutionCSV(database, queryFile);
	
	WaveformFragments queryFragments = solver.makeQueryFragments(database, queryFile);
	WaveformFragments targetFragments = solver.makeTargetFragments(database, queryFile, targetSpeaker);
	queryFragments.sourceSound = queryFile;
	
	queryFragments.align(database);
	//queryFragments.vizAllTransitions();

	targetFragments.align(database);
	//targetFragments.vizAllTransitions();

	solver.dumpSolutionSoundFile(database, queryFragments, "query", queryFile);
	solver.dumpSolutionSoundFile(database, targetFragments, "target", queryFile);

	vector<short> queryWaveform = queryFragments.makeWaveform(database);
	vector<short> targetWaveform = targetFragments.makeWaveform(database);

	queryFragments.saveWAVs(database, querySpeaker, "query");
	targetFragments.saveWAVs(database, querySpeaker, "target");

	cout << "target waveform size: " << targetWaveform.size() << endl;
	SpeechUtil::saveWAVFile(speechParams().debugDir + "queryStitched.wav", queryWaveform);
	SpeechUtil::saveWAVFile(speechParams().debugDir + "targetStitched.wav", targetWaveform);

	//SpeechUtil::vizLoudness(speechParams().debugDir + "queryALoudness.csv", queryFile.waveform);
	//SpeechUtil::vizLoudness(speechParams().debugDir + "queryBLoudness.csv", queryWaveform);
	//SpeechUtil::vizLoudness(speechParams().debugDir + "targetBLoudness.csv", targetWaveform);

	if (targetSpeaker.hasFile(testFileIndex))
	{
		const SoundFile targetFile = targetSpeaker.getFile(testFileIndex).clone();
		const string targetFilename = speechParams().debugDir + "target";
		targetFile.saveWAV(targetFilename);
		targetFile.saveFeatureImage(targetFilename, true);
		SpeechUtil::vizLoudness(speechParams().debugDir + "targetALoudness.csv", targetFile.waveform);
	}

	//solver.dumpSolutionSoundFileQuery(queryFile, querySpeaker);
	//solver.dumpSolutionSoundFileTarget(queryFile, querySpeaker, targetSpeaker);
	
}

void App::testSolver(const FeatureDesc &localFeatureDesc)
{
	const vector<float> baseCosts = { 0.2f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f, 10.0f};
	//const vector<float> baseCosts = { 5.0f };
	for (auto &v : iterate(baseCosts))
	{
		speechParamsMutable().debugDir = speechParams().rootDebugDir + "j-" + to_string(v.value) + "/";
		util::makeDirectory(speechParamsMutable().debugDir);

		SolverParams params;
		params.jumpCost = v.value;
		params.skipCost = params.jumpCost * Constants::skipScaleFactor;
		params.stallCost = params.jumpCost * Constants::skipScaleFactor;
		testSolver(params, localFeatureDesc);
	}
}

void App::makeClusterFile(const string &folderName, const string &outBaseFilename)
{
	auto files = Directory::enumerateFiles(folderName, ".wav");
	sort(files.begin(), files.end());

	ofstream csvFile(outBaseFilename + ".csv");
	csvFile << "index,filename,startSample,endSample" << endl;

	vector<short> separatorSamples = SpeechUtil::readWAVFile(R"(D:\speech2speech\clipDatabase\waterDrop.wav)");
	for (short &s : separatorSamples) s *= 0.25;

	const int bufferSamples = 9000;
	vector<short> outSamples;
	for (int i = 0; i < bufferSamples; i++) outSamples.push_back(0);

	for (int i = 0; i < (int)files.size(); i++)
	{
		const vector<short> curSamples = SpeechUtil::readWAVFile(folderName + files[i]);
		csvFile << i << "," << util::removeExtensions(files[i]) << "," << outSamples.size() << "," << outSamples.size() + curSamples.size() << endl;
		
		for (short s : curSamples) outSamples.push_back(s);
		for (int i = 0; i < bufferSamples; i++) outSamples.push_back(0);
		for (short s : separatorSamples) outSamples.push_back(s);
		for (int i = 0; i < bufferSamples; i++) outSamples.push_back(0);
	}

	SpeechUtil::saveWAVFile(outBaseFilename + ".wav", outSamples);
}

void App::purgeBlacklist()
{
	vector<string> blacklist;
	for (auto &line : util::getFileLines(speechParams().databaseDir + "blacklist.txt", 3))
	{
		blacklist.push_back(line);
	}
	for (auto &filename : Directory::getFilesRecursive(speechParams().databaseDir + "obama"))
	{
		//if (util::endsWith(filename, ".wav")) continue;
		bool isBlacklisted = false;
		for (const string &s : blacklist)
			if (util::contains(filename, s))
				isBlacklisted = true;
		if (isBlacklisted)
		{
			cout << "Deleting " << filename << endl;
			util::deleteFile(filename);
		}
	}
}

void App::createObamaClips()
{
	SoundFile sound;
	VirtualDirectory vDirA, vDirB;
	for (const string &file : Directory::enumerateFiles(speechParams().databaseDir + "obama/WAVNorm/"))
	{
		const string baseFilename = util::removeExtensions(file);
		sound.load(vDirA, vDirB, "obama", baseFilename);

		/*ofstream file("debug.txt");
		for (int frame = 0; frame < sound.frameCount; frame++)
		{
			file << sound.frameAvgValue(frame) << endl;
		}
		return;*/

		SpeechUtil::makeSpeechClips(sound, speechParams().databaseDir + "obama/clips/", sound.baseFilename);
	}
}

void App::alignAllSpeakers()
{
	vector<string> speakers;
	speakers.push_back("bdl");
	speakers.push_back("clb");
	//speakers.push_back("jmk");
	speakers.push_back("rms");
	speakers.push_back("slt");

	for (const string &speakerA : speakers)
	{
		for (const string &speakerB : speakers)
		{
			if (speakerA != speakerB)
				database.alignSpeakers(speakerA, speakerB);
		}
	}
}

void App::runEvaluations()
{
	FeatureEvaluator evaluator;
	evaluator.evaluate(database, speechParams().querySpeakerName, FeatureDesc::defaultJoint());
}

void App::makeDWAVs()
{
	vector<string> speakers;
	speakers.push_back("f9");
	
	for (const string &speaker : speakers)
	{
		const string dirIn = speechParams().databaseDir + speaker + "/WAVNorm/";
		const string dirOut = speechParams().databaseDir + speaker + "/DWAV-CSV/";
		util::makeDirectory(dirOut);
		for (const string &baseFilename : Directory::enumerateFiles(dirIn))
		{
			const string filenameOut = dirOut + util::replace(baseFilename, ".wav", ".csv");
			if (util::fileExists(filenameOut))
			{
				cout << "skipping " << baseFilename << endl;
			}
			else
			{
				cout << "exporting " << baseFilename << endl;
				const vector<int> dwav = SpeechUtil::convertWAVToDWAV(dirIn + baseFilename);
				SpeechUtil::vectorToCSV(dwav, filenameOut);
			}
		}
	}
}

void App::makeTones()
{
	for (string filename : Directory::enumerateFilesWithPath(R"(D:\speech2speech\simpleCorpusC\tone\WAVNorm\)", ".wav"))
	{
		vector<short> w = SpeechUtil::readWAVFile(filename);
		for (auto it : iterate(w))
		{
			it.value = util::boundToShort(sin(it.index * 0.1) * 8000.0);
		}
		SpeechUtil::saveWAVFile(filename, w);
	}
}

void convertAllDWAVs(const string &dir)
{
	for (auto &f : Directory::enumerateFilesWithPath(dir))
	{
		auto fBase = util::removeExtensions(f);
		if (util::contains(fBase, "-final") ||
			util::contains(fBase, "-flat") ||
			util::contains(fBase, "original"))
		{
			SpeechUtil::CSVToWAV(fBase);
		}
	}
}

void mutateWAV(const string &waveIn, const string &waveOut)
{
	auto wIn = SpeechUtil::readWAVFile(waveIn);

	// echoA
	//int echoAOffset = 300;
	//float echoAScale = 0.8f;

	// noiseA
	int echoAOffset = 300;
	float echoAScale = 0.0f;
	float noiseMagnitude = 100.0f;

	float globalScale = 1.0f / (1.0f + echoAScale);

	auto wOut = wIn;
	for (auto &o : iterate(wOut))
	{
		float vOut = o.value;
		vOut += util::clampedRead(wIn, o.index + echoAOffset) * echoAScale;
		vOut += util::randomUniform(-noiseMagnitude, noiseMagnitude);
		vOut *= globalScale;
		o.value = util::boundToShort(vOut);
	}

	SpeechUtil::saveWAVFile(waveOut, wOut);
}

void App::go()
{
	/*for (auto &file : Directory::enumerateFilesWithPath(speechParams().databaseDir + speechParams().targetSpeakerName + "/features-spec-CSV", ".csv"))
	{
		SpeechUtil::addCSVHeaders(file);
	}
	return;*/

	/*for (int i = 1; i <= 5; i++)
	{
		string base = R"(D:\speech2speech\DAPSCorpusA\f9\WAVNorm\f9_script)" + to_string(i);
		mutateWAV(base + "_clean.wav", base + "_noiseA.wav");
	}*/

	//makeTones();
	//return;

	convertAllDWAVs(R"(\\devbox4\matfishe\code\speech2speech\noiseNet\predictions\noiseNetB\)");
	convertAllDWAVs(R"(\\devbox4\matfishe\code\speech2speech\noiseNet\predictions\noiseNetC\)");
	//SpeechUtil::CSVToWAV(R"(\\devbox4\matfishe\code\speech2speech\waveNet\predictions\original)");
	//SpeechUtil::CSVToWAV(R"(\\devbox4\matfishe\code\speech2speech\waveNet\predictions\waveNetF\pred-s80-final)");
	//SpeechUtil::CSVToWAV(R"(\\devbox4\matfishe\code\speech2speech\waveNet\predictions\waveNetF\pred-s80-flat)");
	//SpeechUtil::CSVToWAV(R"(\\devbox4\matfishe\code\speech2speech\waveNet\predictions\waveNetE\pred-s20-final)");
	//SpeechUtil::CSVToWAV(R"(\\devbox4\matfishe\code\speech2speech\waveNet\predictions\waveNetE\pred-s20-flat)");
	//SpeechUtil::CSVToWAV(R"(\\devbox4\matfishe\code\speech2speech\waveNet\predictions\waveNetA\pred-s20-final)");
	//SpeechUtil::CSVToWAV(R"(\\devbox4\matfishe\code\speech2speech\waveNet\predictions\waveNetA\pred-s20-flat)");
	//return;
	
	//SpeechUtil::splitSpeakerParts(R"(D:\speech2speech\faceCorpusAll\matt\WAVAll\)", "matt");
	//SpeechUtil::reduceDeepFeatures("bdl");
	//SpeechUtil::reduceDeepFeatures("obama");
	//SpeechUtil::testFLANN("debug.csv");

	//makeClusterFile(R"(D:\speech2speech\clipDatabase\obama\cluster0\)", R"(D:\speech2speech\clipDatabase\obama\cluster0)");
	//purgeBlacklist();
	//createObamaClips();
	//normalizeAllWAVs();

	//makeDWAVs();

	return;

	//SpeechUtil::testDWAV(speechParams().databaseDir + "rms/WAVNorm/arctic_a0001.wav", speechParams().databaseDir + "rms/DWAV/arctic_a0001.csv");
	//SpeechUtil::convertWAVToDWAV(speechParams().databaseDir + "rms/WAVNorm/arctic_a0001.wav", speechParams().databaseDir + "rms/DWAV/arctic_a0001.dwav");
	//SpeechUtil::convertDWAVToWAV(speechParams().databaseDir + "rms/DWAV/arctic_a0001.dwav", speechParams().databaseDir + "rms/DWAV/arctic_a0001_out.wav");

	database.init();

	return;

	//runEvaluations();
	//return;

	//alignAllSpeakers();

	vector<const SpeakerEntry *> acceleratorSpeakers;

	const bool querySpeakerOnly = false;
	const bool targetSpeakerOnly = true;
	const bool includeQuerySpeaker = false;
	const bool includeTargetSpeaker = true;
	if (querySpeakerOnly)
	{
		acceleratorSpeakers.push_back(database.speakers[speechParams().querySpeakerName]);
	}
	else if (targetSpeakerOnly)
	{
		acceleratorSpeakers.push_back(database.speakers[speechParams().targetSpeakerName]);
	}
	else
	{
		for (const auto &s : database.speakers)
		{
			if (!includeQuerySpeaker && s.second->speakerName == speechParams().querySpeakerName)
				continue;
			if(!includeTargetSpeaker && s.second->speakerName == speechParams().targetSpeakerName)
				continue;
			acceleratorSpeakers.push_back(s.second);
		}
	}
	//acceleratorSpeakers.push_back(&database.getSpeaker("obama"));

	const FeatureDesc desc = FeatureDesc::defaultJoint();

	accelerator.init(database, acceleratorSpeakers, database.getSpeaker(speechParams().targetSpeakerName), Constants::kNearest);

	//accelerator.init(*database.speakers["obama"], Constants::kNearest, Constants::descriptorSampleLength, Constants::compactDescriptorTerms, SoundFeatureType::Joint);

	//testAccelerator();
	testSolver(desc);
}
