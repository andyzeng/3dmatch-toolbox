// Source file for PDB distance computations


// Include files

#include "PDB.h"



RNLength 
PDBDistance(PDBAtom *atom1, PDBAtom *atom2)
{
  // Return distance between two atoms
  RNLength distance = R3Distance(atom1->Position(), atom2->Position());
  distance -= atom1->Radius();
  distance -= atom2->Radius();
  return (distance > 0) ? distance : 0;
}



RNLength 
PDBDistance(PDBAtom *atom1, PDBResidue *residue)
{
  // Initialize distance
  RNLength distance = RN_INFINITY;

  // Find distance from atom1 to closest atom in residue
  for (int i = 0; i < residue->NAtoms(); i++) {
    PDBAtom *atom2 = residue->Atom(i);
    RNLength d = PDBDistance(atom1, atom2);
    if (d < distance) distance = d;
  }

  // Return distance between atom and residue
  return distance;
}



RNLength 
PDBDistance(PDBAtom *atom, PDBChain *chain)
{
  // Initialize distance
  RNLength distance = RN_INFINITY;

  // Find distance from atom to closest atom in chain
  for (int i = 0; i < chain->NResidues(); i++) {
    PDBResidue *residue = chain->Residue(i);
    if (R3Distance(atom->Position(), residue->BBox()) < distance) {
      RNLength d = PDBDistance(atom, residue);
      if (d < distance) distance = d;
    }
  }

  // Return distance between atom and chain
  return distance;
}



RNLength 
PDBDistance(PDBResidue *residue1, PDBResidue *residue2)
{
  // Initialize distance
  RNLength distance = RN_INFINITY;

  // Find least distance from an atom in residue1 to an atom in residue2
  for (int i = 0; i < residue1->NAtoms(); i++) {
    PDBAtom *atom1 = residue1->Atom(i);
    if (R3Distance(atom1->Position(), residue2->BBox()) < distance) {
      RNLength d = PDBDistance(atom1, residue2);
      if (d < distance) distance = d;
    }
  }

  // Return distance between residues
  return distance;
}



RNLength 
PDBDistance(PDBResidue *residue1, PDBChain *chain)
{
  // Initialize distance
  RNLength distance = RN_INFINITY;

  // Find distance between closest atoms from residue and chain
  for (int i = 0; i < chain->NResidues(); i++) {
    PDBResidue *residue2 = chain->Residue(i);
    if (R3Distance(residue1->BBox(), residue2->BBox()) < distance) {
      RNLength d = PDBDistance(residue1, residue2);
      if (d < distance) distance = d;
    }
  }

  // Return distance between residue and chain
  return distance;
}





RNLength 
PDBDistance(PDBChain *chain1, PDBChain *chain2)
{
  // Initialize distance
  RNLength distance = RN_INFINITY;

  // Find least distance from an atom in chain1 to an atom in chain2
  for (int i = 0; i < chain1->NResidues(); i++) {
    PDBResidue *residue1 = chain1->Residue(i);
    if (R3Distance(residue1->BBox(), chain2->BBox()) < distance) {
      RNLength d = PDBDistance(residue1, chain2);
      if (d < distance) distance = d;
    }
  }

  // Return distance between chains
  return distance;
}




