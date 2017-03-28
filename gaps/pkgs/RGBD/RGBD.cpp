/* Source file for GAPS rgbd module */



/* Include files */

#include "RGBD.h"



/* Private variables */

static int RGBD_active_count = 0;



int R2InitShapes(void)
{
    // Check whether are already initialized 
    if ((RGBD_active_count++) > 0) return TRUE;

    // Initialize dependencies
    if (!R3InitShapes()) return FALSE;

    // return OK status 
    return TRUE;
}



void R2StopShapes(void)
{
    // Check whether have been initialized 
    if ((--RGBD_active_count) > 0) return;

    // Stop dependencies
    R3StopShapes();
}






