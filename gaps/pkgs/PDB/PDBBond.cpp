// Source file for PDBElement



// Include files

#include "PDB.h"



PDBBond::
PDBBond(PDBAtom *atom1, PDBAtom *atom2, PDBBondType type)
  : type(type),
    mark(0),
    value(0),
    data(NULL)
{
  // Just checking
  assert(atom1 && atom2);

  // Assign atoms
  atoms[0] = atom1;
  atoms[1] = atom2;

  // Assign ID
  static int PDBnext_bond_id = 0;
  id = PDBnext_bond_id++;

  // Insert bond into atom1, residue1, chain1, and model1
  atom1->bonds.Insert(this); 
  PDBResidue *residue1 = atom1->Residue();
  if (residue1) {
    residue1->bonds.Insert(this);
    PDBChain *chain1 = residue1->Chain();
    if (chain1) {
      chain1->bonds.Insert(this);
      PDBModel *model1 = chain1->Model();
      if (model1) {
        model1->bonds.Insert(this);
      }
    }
  }

  // Insert bond into atom2, residue2, chain2, and model2
  atom2->bonds.Insert(this); 
  PDBResidue *residue2 = atom2->Residue();
  if (residue2) {
    residue2->bonds.Insert(this);
    PDBChain *chain2 = residue2->Chain();
    if (chain2) {
      chain2->bonds.Insert(this);
      PDBModel *model2 = chain2->Model();
      if (model2) {
        model2->bonds.Insert(this);
      }
    }
  }
}



PDBBond::
~PDBBond(void)
{
  // Remove bond from atom1, residue1, chain1, and model1
  atoms[0]->bonds.Remove(this); 
  PDBResidue *residue1 = atoms[0]->Residue();
  if (residue1) {
    residue1->bonds.Remove(this);
    PDBChain *chain1 = residue1->Chain();
    if (chain1) {
      chain1->bonds.Remove(this);
      PDBModel *model1 = chain1->Model();
      if (model1) {
        model1->bonds.Remove(this);
      }
    }
  }

  // Remove bond from atom2, residue2, chain2, and model2
  atoms[1]->bonds.Remove(this); 
  PDBResidue *residue2 = atoms[1]->Residue();
  if (residue2) {
    residue2->bonds.Remove(this);
    PDBChain *chain2 = residue2->Chain();
    if (chain2) {
      chain2->bonds.Remove(this);
      PDBModel *model2 = chain2->Model();
      if (model2) {
        model2->bonds.Remove(this);
      }
    }
  }
}



const RNRgb& PDBBond::
Color(void) const
{
  // Return color indicated by bond type
  static const RNRgb colors[PDB_NUM_BOND_TYPES] = { 
    RNRgb(0.3, 0.2, 0.1), RNRgb(1, 0, 0), RNRgb(0, 0, 1), RNRgb(1, 1, 1) 
  };
  return colors[type];
}



