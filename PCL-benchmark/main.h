
#pragma once

//#define USE_MATLAB

#include "mLibInclude.h"

#include "benchmarkParams.h"
#include "benchmarkUtil.h"

#include "pclWrapperFPFH.h"

#include "app.h"

extern BenchmarkParams* g_benchmarkParams;
inline const BenchmarkParams& benchmarkParams()
{
	return *g_benchmarkParams;
}

inline BenchmarkParams& benchmarkParamsMutable()
{
	return *g_benchmarkParams;
}
