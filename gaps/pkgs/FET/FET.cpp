/* Source file for FET module */



/* Include files */

#include "FET.h"



/* Private variables */

static int FET_active_count = 0;



int R3InitFET(void)
{
    // Check whether are already initialized 
    if ((FET_active_count++) > 0) return TRUE;

    // Initialize dependencies
    if (!R3InitShapes()) return FALSE;

    // return OK status 
    return TRUE;
}



void R3StopFET(void)
{
    // Check whether have been initialized 
    if ((--FET_active_count) > 0) return;

    // Stop dependencies
    R3StopShapes();
}




