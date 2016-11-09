
#include "main.h"

KeypointParameters *g_keypointParams;

void main()
{
	g_keypointParams = new KeypointParameters();

	App app;
	app.go();
	
	cout << "Done!" << endl;
	cin.get();
}
