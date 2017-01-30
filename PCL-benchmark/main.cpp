
#include "main.h"

BenchmarkParams *g_benchmarkParams;

void convertDirectory(const string &dir)
{
	for (auto & filename : Directory::enumerateFiles(dir, ".pcd"))
	{
		const string command = "pcl_pcd2ply_release " + dir + filename + " " + dir + util::replace(filename, ".pcd", ".ply");
		system(command.c_str());
	}
}

void main()
{
	g_benchmarkParams = new BenchmarkParams();

	convertDirectory(R"(C:\Users\matfishe\Downloads\formatt\fr2-xyz\pcds\)");
	convertDirectory(R"(C:\Users\matfishe\Downloads\formatt\fr2-xyz-old\pcds\)");
	convertDirectory(R"(C:\Users\matfishe\Downloads\formatt\fr3-nstn\pcds\)");
	convertDirectory(R"(C:\Users\matfishe\Downloads\formatt\fr3-office\pcds\)");
	
	return;

	App app;
	app.go();
	
	cout << "Done!" << endl;
	cin.get();
}
