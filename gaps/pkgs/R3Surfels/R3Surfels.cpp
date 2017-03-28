/* Source file for GAPS surfels module */



/* Include files */

#include "R3Surfels.h"



/* Private variables */

static int R3surfels_active_count = 0;



int R3InitSurfels(void)
{
    // Check whether are already initialized 
    if ((R3surfels_active_count++) > 0) return TRUE;

    // Initialize dependencies
    if (!R3InitShapes()) return FALSE;

    // return OK status 
    return TRUE;
}



void R3StopSurfels(void)
{
    // Check whether have been initialized 
    if ((--R3surfels_active_count) > 0) return;

    // Stop dependencies
    R3StopShapes();
}




