// Include file for debug functions

int DebugRedraw(R3SurfelViewer *viewer);
int DebugResize(R3SurfelViewer *viewer, int w, int h);
int DebugMouseMotion(R3SurfelViewer *viewer, int x, int y);
int DebugMouseButton(R3SurfelViewer *viewer, int x, int y, int button, int state, int shift, int ctrl, int alt);
int DebugKeyboard(R3SurfelViewer *viewer, int x, int y, int key, int shift, int ctrl, int alt);

