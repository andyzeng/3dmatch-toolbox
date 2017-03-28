/* Source file for PDB package */



/* Include files */

#include "PDB.h"



// Global variables 

RNMark PDBmark = 1;



/* Private variables */

static int PDBactive_count = 0;



int PDBInit(void)
{
  // Check whether are already initialized 
  if ((PDBactive_count++) > 0) return TRUE;

  // Initialize submodules 
  // ???

  // Return OK status 
  return TRUE;
}



void PDBStop(void)
{
  // Check whether have been initialized 
  if ((--PDBactive_count) > 0) return;

  // Stop submodules 
  // ???
}



void PDBClearMarks(void)
{
  // Create new mark
  PDBmark++;
}
