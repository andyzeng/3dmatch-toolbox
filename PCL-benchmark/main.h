
#pragma once

//#define USE_MATLAB

#include "mLibInclude.h"
#include "virtualDirectory.h"

#include "enums.h"
#include "speechParams.h"
#include "speechUtil.h"
#include "soundFile.h"
#include "soundDatabase.h"
#include "speechAccelerator.h"
#include "speechAcceleratorEnsemble.h"

#include "waveformFragments.h"
#include "waveformSolver.h"
#include "dynamicTimeWarping.h"
#include "featureEvaluator.h"

#include "app.h"

extern SpeechParameters* g_speechParams;
inline const SpeechParameters& speechParams()
{
	return *g_speechParams;
}

inline SpeechParameters& speechParamsMutable()
{
	return *g_speechParams;
}
