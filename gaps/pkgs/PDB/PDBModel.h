// Header file for PDBModel class



// Class declaration

class PDBModel {
public:
  // Constructor
  PDBModel(PDBFile *file, const char *name = NULL);
  ~PDBModel(void);

  // Property functions
  int ID(void) const;
  const char *Name(void) const;
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;
  RNLength Radius(void) const;
  const RNRgb& Color(void) const;
  RNBoolean IsMarked(void) const;
  RNScalar Value(void) const;
  void *Data(void) const;

  // Atom access functions
  int NAtoms(void) const;
  PDBAtom *Atom(int k) const;

  // Residue access functions
  int NResidues(void) const;
  PDBResidue *Residue(int k) const;

  // Chain access functions
  int NChains(void) const;
  PDBChain *Chain(int k) const;

  // File access functions
  PDBFile *File(void) const;

  // Bond access functions
  int NBonds(void) const;
  PDBBond *Bond(int k) const;

  // Search access functions
  PDBAtom *FindAtom(const char *str) const;
  PDBResidue *FindResidue(const char *str) const;
  PDBChain *FindChain(const char *str) const;
  PDBStructureType FindAny(const char *str, 
    PDBChain **chain = NULL, PDBResidue **residue = NULL, 
    PDBAtom **atom = NULL, PDBStructureType maxlevel = PDB_ATOM) const;

  // Manipulation functions
  void Transform(const R3Affine& affine);
  PDBChain *InsertCopy(PDBChain *chain);
  void SetName(const char *name);
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

public:
  // Data fields
  int id;
  char name[32];
  R3Box bbox;
  RNArray<PDBAtom *> atoms;
  RNArray<PDBResidue *> residues;
  RNArray<PDBChain *> chains;
  RNArray<PDBBond *> bonds;
  PDBFile *file;
  RNMark mark;
  RNScalar value;
  void *data;
};



// Inline functions

inline int PDBModel::
ID(void) const
{
  // Return id
  return id;
}



inline const char *PDBModel::
Name(void) const
{
  // Return name
  return name;
}



inline const R3Box& PDBModel::
BBox(void) const
{
  // Return bounding box (including atom radii)
  return bbox;
}



inline R3Point PDBModel::
Centroid(void) const
{
  // Return center of mass
  return PDBCentroid(atoms);
}



inline RNLength PDBModel::
Radius(void) const
{
  // Return max distance to centroid
  return PDBMaxDistance(atoms, Centroid());
}



inline RNMark PDBModel::
IsMarked(void) const
{
  // Return whether model is marked
  return (mark == PDBmark);
}



inline RNScalar PDBModel::
Value(void) const
{
  // Return scalar value associated with model by user (this is not used by the PDB package)
  return value;
}



inline void *PDBModel::
Data(void) const
{
  // Return data pointer associated with model by user (this is not used by the PDB package)
  return data;
}



inline int PDBModel::
NAtoms(void) const
{
  // Return number of atoms
  return atoms.NEntries();
}



inline PDBAtom *PDBModel::
Atom(int k) const
{
  // Return Kth atom
  return atoms.Kth(k);
}



inline int PDBModel::
NResidues(void) const
{
  // Return number of residues
  return residues.NEntries();
}



inline PDBResidue *PDBModel::
Residue(int k) const
{
  // Return Kth residue
  return residues.Kth(k);
}


inline int PDBModel::
NChains(void) const
{
  return chains.NEntries();
}



inline PDBChain *PDBModel::
Chain(int k) const
{
  return chains.Kth(k);
}



inline int PDBModel::
NBonds(void) const
{
  // Return number of bonds
  return bonds.NEntries();
}



inline PDBBond *PDBModel::
Bond(int k) const
{
  // Return Kth bond
  return bonds.Kth(k);
}



inline PDBFile *PDBModel::
File(void) const
{
  return file;
}



inline void PDBModel::
SetName(const char *name)
{
  // Set name
  strncpy(this->name, name, 32);
  this->name[31] = '\0';
}



inline void PDBModel::
SetData(void *data)
{
  // Set data pointer associated with model by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBModel::
SetValue(RNScalar value)
{
  // Set scalar value associated with model by user (this is not used by the PDB package)
  this->value = value;
}



inline void PDBModel::
SetMark(void) 
{
  // Mark this model
  mark = PDBmark;
}



inline void PDBModel::
UnsetMark(void) 
{
  // Unmark this model
  mark = 0;
}



inline PDBAtom *PDBModel:: 
FindAtom(const char *str) const
{
  // Find atom matching string
  PDBAtom *atom = NULL;
  if (FindAny(str, NULL, NULL, &atom, PDB_ATOM) == PDB_ATOM) return atom;
  else return NULL;
}



inline PDBResidue *PDBModel:: 
FindResidue(const char *str) const
{
  // Find residue matching string
  PDBResidue *residue = NULL;
  if (FindAny(str, NULL, &residue, NULL, PDB_RESIDUE) == PDB_RESIDUE) return residue;
  else return NULL;
}



inline PDBChain *PDBModel:: 
FindChain(const char *str) const
{
  // Find chain matching string
  PDBChain *chain = NULL;
  if (FindAny(str, &chain, NULL, NULL, PDB_CHAIN) == PDB_CHAIN) return chain;
  else return NULL;
}



