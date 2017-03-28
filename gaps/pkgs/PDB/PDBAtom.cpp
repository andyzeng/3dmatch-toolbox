// Source file for PDBAtom



// Include files

#include "PDB.h"



PDBAtom::
PDBAtom(PDBModel *model, PDBChain *chain, PDBResidue *residue, PDBElement *element,
  int serial, const char *atom_name, int altLoc, RNScalar x, RNScalar y, RNScalar z, 
  RNScalar occupancy, RNScalar tempFactor, RNScalar charge, RNBoolean hetatm)
  : serial(serial),
    altLoc(altLoc),
    position(x, y, z),
    occupancy(occupancy),
    tempFactor(tempFactor),
    charge(charge),
    hetatm(hetatm),
    residue(residue),
    element(element),
    accessible_surface_area(PDB_UNKNOWN),
    aminoacid_atom_type(PDB_NULL_ATOM),
    mark(0),
    value(0),
    data(NULL)
{
  // Just checking
  assert(!model || !chain || (chain->model == model));
  assert(!chain || !residue || (residue->chain == chain));

  // Assign ID
  static int PDBnext_atom_id = 0;
  id = PDBnext_atom_id++;

  // Copy name
  if (atom_name) { strncpy(this->name, atom_name, 8); this->name[7] = 0; }
  else this->name[0] = '\0';

  // Determine atom bounding box
  RNLength radius = Radius();
  R3Vector half_diagonal(radius, radius, radius);
  R3Box bbox(position - half_diagonal, position + half_diagonal);

  // Determine aminoacid atom type
  aminoacid_atom_type = FindAminoAcidAtomType(this);

  // Insert atome into residue, chain, and model
  if (residue) { residue->atoms.Insert(this); residue->bbox.Union(bbox); }
  if (chain) { chain->atoms.Insert(this); chain->bbox.Union(bbox); }
  if (model) { model->atoms.Insert(this); model->bbox.Union(bbox); }
}



PDBAtom::
~PDBAtom(void)
{
  // Delete bonds
  while (NBonds()) delete bonds.Tail();

  // Remove atom from residue, chain, and model
  if (residue) {
    residue->atoms.Remove(this);
    PDBChain *chain = residue->Chain();
    if (chain) {
      chain->atoms.Remove(this);
      PDBModel *model = chain->Model();
      if (model) {
        model->atoms.Remove(this);
      }
    }
  }
}



PDBAminoAcid *PDBAtom::
AminoAcid(void) const
{
  // Return aminoacid
  return (residue) ? residue->AminoAcid() : NULL;
}



PDBChain *PDBAtom::
Chain(void) const
{
  // Return chain
  return (residue) ? residue->Chain() : NULL;
}



PDBModel *PDBAtom::
Model(void) const
{
  // Return chain
  PDBChain *chain = Chain();
  return (chain) ? chain->Model() : NULL;
}



PDBFile *PDBAtom::
File(void) const
{
  // Return chain
  PDBModel *model = Model();
  return (model) ? model->File() : NULL;
}



RNBoolean PDBAtom::
IsBonded(PDBAtom *atom2)
{
  // Search for bond to atom2
  for (int i = 0; i < NBonds(); i++) {
    PDBBond *bond = Bond(i);
    if (bond->OtherAtom(this) == atom2) return TRUE;
  }

  // Bond not found
  return FALSE;
}



RNBoolean PDBAtom::
IsBonded(PDBResidue *residue2)
{
  // Search for bond to any atom in residue2
  for (int i = 0; i < NBonds(); i++) {
    PDBBond *bond = Bond(i);
    if (bond->OtherAtom(this)->Residue() == residue2) return TRUE;
  }

  // Bond not found
  return FALSE;
}



void PDBAtom::
SetPosition(const R3Point& position)
{
  // Set the atom's position
  this->position = position;

  // Determine atom bounding box
  RNLength radius = Radius();
  R3Vector half_diagonal(radius, radius, radius);
  R3Box bbox(position - half_diagonal, position + half_diagonal);

  // Update bounding boxes of residue, chain, and model
  if (residue) { 
    residue->bbox.Union(bbox); 
    PDBChain *chain = residue->chain;
    if (chain) { 
      chain->bbox.Union(bbox); 
      PDBModel *model = chain->model;
      if (model) model->bbox.Union(bbox); 
    }
  }
}



void PDBAtom::
Transform(const R3Affine& affine)
{
  // Set the atom's position
  position.Transform(affine);

  // Determine atom bounding box
  RNLength radius = Radius();
  R3Vector half_diagonal(radius, radius, radius);
  R3Box bbox(position - half_diagonal, position + half_diagonal);

  // Update bounding boxes of residue, chain, and model
  if (residue) { 
    residue->bbox.Union(bbox); 
    PDBChain *chain = residue->chain;
    if (chain) { 
      chain->bbox.Union(bbox); 
      PDBModel *model = chain->model;
      if (model) model->bbox.Union(bbox); 
    }
  }
}



