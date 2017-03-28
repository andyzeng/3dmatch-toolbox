/* Source file for GAPS math module */



/* Include files */

#include "RNMath.h"



/* Private variables */

static int RNmath_active_count = 0;



int RNInitMath(void)
{
    // Check whether are already initialized 
    if ((RNmath_active_count++) > 0) return TRUE;

    // Initialize submodules 
    RNSeedRandomScalar();

    // Return OK status 
    return TRUE;
}



void RNStopMath(void)
{
    // Check whether have been initialized 
    if ((--RNmath_active_count) > 0) return;

    // Stop submodules 
    // ???
}




