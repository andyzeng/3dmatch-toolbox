// Header file for PDB package

#ifndef __PDB__H__
#define __PDB__H__



// Dependency include files

#include "R3Shapes/R3Shapes.h"



// PDB structure class pre-declarations

class PDBElement;
class PDBAminoAcid;
class PDBAtom;
class PDBResidue;
class PDBChain;
class PDBModel;
class PDBFile;
class PDBBond;



// PDB structure types

typedef enum {
  PDB_FILE,
  PDB_MODEL,
  PDB_CHAIN,
  PDB_RESIDUE,
  PDB_ATOM
} PDBStructureType;



// Global variables 

extern RNMark PDBmark;



// Useful constants

#define PDB_UNKNOWN (123456789E-10)



// PDB structure include files

#include "PDBUtil.h"
#include "PDBAtomTypes.h"
#include "PDBElement.h"
#include "PDBAminoAcid.h"
#include "PDBAtom.h"
#include "PDBResidue.h"
#include "PDBChain.h"
#include "PDBModel.h"
#include "PDBFile.h"
#include "PDBBond.h"
#include "PDBDistance.h"



// Initialization functions 

int PDBInit(void);
void PDBStop(void);



// Other functions

void PDBClearMarks(void);



#endif







