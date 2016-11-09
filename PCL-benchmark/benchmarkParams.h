
struct Constants
{
	static const int a = 160;
	static constexpr float b = 5.0f;
};

struct BenchmarkParams
{
	BenchmarkParams()
	{
		outDir = R"(D:\speech2speech\DAPSCorpusA\)";
	}

	string outDir;
};
