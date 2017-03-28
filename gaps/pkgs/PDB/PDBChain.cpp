// Source file for PDBChain class



// Include files

#include "PDB.h"



PDBChain::
PDBChain(PDBModel *model, const char *name)
  : bbox(R3null_box),
    model(model),
    mark(0),
    value(0),
    data(NULL)
{
  // Assign ID
  static int PDBnext_chain_id = 0;
  id = PDBnext_chain_id++;

  // Copy name
  if (name) { strncpy(this->name, name, 4); this->name[3] = 0; }
  else this->name[0] = '\0';

  // Insert chain into model
  if (model) model->chains.Insert(this);
}



PDBChain::
~PDBChain(void)
{
  // Delete residues
  while (NResidues()) delete residues.Tail();

  // Remove chain from model
  if (model) model->chains.Remove(this);
}



const RNRgb& PDBChain::
Color(void) const
{
  static const RNRgb colors[8] = { 
    RNRgb(0, 0, 1), RNRgb(1, 0, 0), RNRgb(0, 0.7, 0), RNRgb(0, 1, 1), 
    RNRgb(1, 0, 1), RNRgb(1, 0.5, 1), RNRgb(1, 1, 0.5), RNRgb(0.5, 1, 1)
  };
  return colors[id % 8];
}



PDBFile *PDBChain::
File(void) const
{
  // Return file this chain is part of
  return (model) ? model->File() : NULL;
}



PDBResidue *PDBChain::
FindResidue(const char *residue_name, int residue_sequence, int residue_insertion_code) const
{
  // Check all residues
  for (int i = 0; i < NResidues(); i++) {
    PDBResidue *residue = Residue(i);
    if ((residue_sequence == residue->Sequence()) &&
        (residue_insertion_code == residue->InsertionCode()) &&
        (!strcmp(residue_name, residue->Name()))) {
      return residue;
    }
  }
  return NULL;
}



PDBStructureType PDBChain::
FindAny(const char *str, PDBResidue **r, PDBAtom **a, PDBStructureType maxlevel) const
{
  // Check string
  if (!str) return PDB_CHAIN;
  if (strlen(str) < 1) return PDB_CHAIN;

  // Copy string
  char buffer[64];
  strncpy(buffer, str, 64);
  buffer[63] = '\0';

  // Parse tokens 
  char *bufferp = buffer;
  char *namep = buffer;
  char *seqp = NULL;
  char *icodep = NULL;
  while (*bufferp) {
    if (*bufferp == '_') *bufferp = ' '; 
    else if ((*bufferp == '-') || (*bufferp == '.')) {
      *(bufferp) = '\0';
      if (!seqp) seqp = bufferp+1;
      else if (!icodep) icodep = bufferp+1;
      else break;
    }
    bufferp++;
  }

  // Find residue
  PDBResidue *result = NULL;
  if (namep && seqp && icodep) {
    char *residue_name = namep;
    int residue_sequence = atoi(seqp);
    int residue_insertion_code = icodep[0];
    result = FindResidue(residue_name, residue_sequence, residue_insertion_code);
  }

  // Fill return values and consider matches at deeper levels
  PDBStructureType level = PDB_CHAIN;
  if (result) {
    level = PDB_RESIDUE;
    if (r) *r = result;
    const char *strp = str + (bufferp - buffer);
    if (*strp && (maxlevel > level)) {
      level = result->FindAny(strp, a, maxlevel);
    }
  }

  // Return level of match 
  return level;
}



void PDBChain::
Transform(const R3Affine& affine)
{
  // Transform all residues by affine
  bbox = R3null_box;
  for (int i = 0; i < residues.NEntries(); i++) 
    residues[i]->Transform(affine);
}



PDBResidue *PDBChain::
InsertCopy(PDBResidue *r)
{
  // Insert copy of residue into this chain
  PDBResidue *residue = new PDBResidue(Model(), this, r->Name(), r->Sequence(), r->InsertionCode());
  assert(residue);

  // Insert copy of all r's atoms
  for (int i = 0; i < r->NAtoms(); i++) 
    residue->InsertCopy(r->Atom(i));

  // Return residue
  return residue;
}



