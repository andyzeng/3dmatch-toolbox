
#include "main.h"

BenchmarkParams *g_benchmarkParams;

void main()
{
	g_benchmarkParams = new BenchmarkParams();

	App app;
	app.go();
	
	cout << "Done!" << endl;
	cin.get();
}
