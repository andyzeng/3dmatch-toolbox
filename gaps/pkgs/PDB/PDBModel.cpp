// Source file for PDBModel structure



// Include files

#include "PDB.h"



PDBModel::
PDBModel(PDBFile *file, const char *name)
  : bbox(R3null_box),
    file(file),
    mark(0),
    value(0),
    data(NULL)
{
  // Assign ID
  static int PDBnext_model_id = 0;
  id = PDBnext_model_id++;

  // Copy name
  if (name) { strncpy(this->name, name, 32); this->name[31] = 0; }
  else this->name[0] = '\0';

  // Add model to file
  if (file) file->models.Insert(this);
}



PDBModel::
~PDBModel(void)
{
  // Delete chains
  while (NChains()) delete chains.Tail();

  // Remove model from file
  if (file) file->models.Remove(this);
}



const RNRgb& PDBModel::
Color(void) const
{
  static const RNRgb colors[8] = { 
    RNRgb(0, 0, 1), RNRgb(1, 0, 0), RNRgb(0, 0.7, 0), RNRgb(0, 1, 1), 
    RNRgb(1, 0, 1), RNRgb(1, 0.5, 1), RNRgb(1, 1, 0.5), RNRgb(0.5, 1, 1)
  };
  return colors[id % 8];
}



PDBStructureType PDBModel::
FindAny(const char *str, PDBChain **c, PDBResidue **r, PDBAtom **a, PDBStructureType maxlevel) const
{
  // Check str
  if (!str) return PDB_MODEL;
  if (strlen(str) < 1) return PDB_MODEL;

  // Construct chain name
  char chain_name[4];
  chain_name[0] = str[0];
  if (chain_name[0] == '_') chain_name[0] = ' ';
  chain_name[1] = '\0';

  // Find chain from name
  PDBChain *result = NULL;
  for (int i = 0; i < NChains(); i++) {
    PDBChain *chain = Chain(i);
    if (!strcmp(chain->Name(), chain_name)) {
      result = chain;
      break;
    }
  }

  // Fill return values and consider matches at deeper levels
  PDBStructureType level = PDB_MODEL;
  if (result) {
    level = PDB_CHAIN;
    if (c) *c = result;
    const char *strp = (str[1] == '-') ? &str[2] : NULL;
    if (strp && *strp && (maxlevel > level)) {
      level = result->FindAny(strp, r, a, maxlevel);
    }
  }

  // Return level of match 
  return level;
}



void PDBModel::
Transform(const R3Affine& affine)
{
  // Transform all chains by affine transformation
  bbox = R3null_box;
  for (int i = 0; i < chains.NEntries(); i++) 
    chains[i]->Transform(affine);
}



PDBChain *PDBModel::
InsertCopy(PDBChain *c)
{
  // Insert copy of chain into this model
  PDBChain *chain =  new PDBChain(this, c->Name());
  assert(chain);

  // Insert copy of all c's residues 
  for (int i = 0; i < c->NResidues(); i++) 
    chain->InsertCopy(c->Residue(i));

  // Return chain
  return chain;
}



