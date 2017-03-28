// Include file for PDBChain classr



// Class declaration

class PDBChain {
public:
  // Constructor
  PDBChain(PDBModel *model, const char *name);
  ~PDBChain(void);

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

  // Other access functions
  PDBModel *Model(void) const;
  PDBFile *File(void) const;

  // Bond access functions
  int NBonds(void) const;
  PDBBond *Bond(int k) const;

  // Search access functions
  PDBAtom *FindAtom(const char *str) const;
  PDBResidue *FindResidue(const char *str) const;
  PDBResidue *FindResidue(const char *residue_name, int residue_sequence, int residue_insertion_code) const;
  PDBStructureType FindAny(const char *str, PDBResidue **residue = NULL, 
    PDBAtom **atom = NULL, PDBStructureType maxlevel = PDB_ATOM) const;

  // Manipulation functions
  void Transform(const R3Affine& affine);
  PDBResidue *InsertCopy(PDBResidue *residue);
  void SetName(const char *name);
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

public:
  // Data fields
  int id;
  char name[4];
  R3Box bbox;
  PDBModel *model;
  RNArray<PDBAtom *> atoms;
  RNArray<PDBResidue *> residues;
  RNArray<PDBBond *> bonds;
  RNMark mark;
  RNScalar value;
  void *data;
};



// Inline functions

inline int PDBChain::
ID(void) const
{
  // Return id
  return id;
}



inline const char *PDBChain::
Name(void) const
{
  // Return name
  return name;
}



inline const R3Box& PDBChain::
BBox(void) const
{
  // Return bounding box (including atom radii)
  return bbox;
}



inline R3Point PDBChain::
Centroid(void) const
{
  // Return center of mass
  return PDBCentroid(atoms);
}



inline RNLength PDBChain::
Radius(void) const
{
  // Return max distance to centroid
  return PDBMaxDistance(atoms, Centroid());
}



inline RNBoolean PDBChain::
IsMarked(void) const
{
  // Return whether chain is marked
  return (mark == PDBmark);
}



inline RNScalar PDBChain::
Value(void) const
{
  // Return scalar value associated with chain by user (this is not used by the PDB package)
  return value;
}



inline void *PDBChain::
Data(void) const
{
  // Return data pointer associated with chain by user (this is not used by the PDB package)
  return data;
}



inline int PDBChain::
NAtoms(void) const
{
  // Return number of atoms
  return atoms.NEntries();
}



inline PDBAtom *PDBChain::
Atom(int k) const
{
  // Return Kth atom
  return atoms.Kth(k);
}



inline int PDBChain::
NResidues(void) const
{
  // Return number of residues
  return residues.NEntries();
}



inline PDBResidue *PDBChain::
Residue(int k) const
{
  // Return Kth residue
  return residues.Kth(k);
}



inline int PDBChain::
NBonds(void) const
{
  // Return number of bonds
  return bonds.NEntries();
}



inline PDBBond *PDBChain::
Bond(int k) const
{
  // Return Kth bond
  return bonds.Kth(k);
}



inline PDBModel *PDBChain::
Model(void) const
{
  // Return model this chain is part of
  return model;
}



inline void PDBChain::
SetName(const char *name)
{
  // Set name
  strncpy(this->name, name, 2);
  this->name[1] = '\0';
}



inline void PDBChain::
SetData(void *data)
{
  // Set data pointer associated with chain by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBChain::
SetValue(RNScalar value)
{
  // Set scalar value associated with chain by user (this is not used by the PDB package)
  this->value = value;
}



inline void PDBChain::
SetMark(void) 
{
  // Mark this chain
  mark = PDBmark;
}



inline void PDBChain::
UnsetMark(void) 
{
  // Unmark this chain
  mark = 0;
}


inline PDBAtom *PDBChain:: 
FindAtom(const char *str) const
{
  // Find atom matching string
  PDBAtom *atom = NULL;
  if (FindAny(str, NULL, &atom, PDB_ATOM) == PDB_ATOM) return atom;
  else return NULL;
}



inline PDBResidue *PDBChain:: 
FindResidue(const char *str) const
{
  // Find residue matching string
  PDBResidue *residue = NULL;
  if (FindAny(str, &residue, NULL, PDB_RESIDUE) == PDB_RESIDUE) return residue;
  else return NULL;
}



