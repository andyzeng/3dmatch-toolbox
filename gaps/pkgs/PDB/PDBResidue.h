// Include file for PDBResidue class


 
// Class declaration

class PDBResidue {
public:
  // Constructor
  PDBResidue(PDBModel *model, PDBChain *chain, const char *name, int sequence, int insertion_code);
  ~PDBResidue(void);

  // Property functions
  int ID(void) const;
  const char *Name(void) const;
  int Sequence(void) const;
  int InsertionCode(void) const;
  const RNRgb& Color(void) const;
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;
  RNLength Radius(void) const;
  RNBoolean HasHetAtoms(void) const;
  RNBoolean IsBonded(PDBAtom *atom);
  RNBoolean IsBonded(PDBResidue *residue);
  RNBoolean IsMarked(void) const;
  RNScalar Value(void) const;
  void *Data(void) const;

  // Atom access functions
  int NAtoms(void) const;
  PDBAtom *Atom(int k) const;

  // Bond access functions
  int NBonds(void) const;
  PDBBond *Bond(int k) const;

  // Other access functions
  PDBAminoAcid *AminoAcid(void) const;
  PDBChain *Chain(void) const;
  PDBModel *Model(void) const;
  PDBFile *File(void) const;

  // Search access functions
  PDBAtom *FindAtom(const char *str) const;
  PDBAtom *FindAtom(const char *atom_name, int atom_serial, int altLoc = 0) const;
  int FindAtoms(char **atom_names, PDBAtom **atom_ptrs, int natoms);
  PDBStructureType FindAny(const char *str, PDBAtom **atom = NULL, PDBStructureType maxlevel = PDB_ATOM) const;

  // Manipulation functions
  void Transform(const R3Affine& affine);
  PDBAtom *InsertCopy(PDBAtom *atom);
  void SetName(const char *name);
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

  // Other stuff
  int ConstructCoordinateSystem(R3CoordSystem& cs, int coordinate_system_type = 0);

public:
  // Data fields
  int id;
  char name[4];
  int sequence;
  int insertion_code;
  R3Box bbox;
  PDBChain *chain;
  PDBAminoAcid *aminoacid;
  RNArray<PDBAtom *> atoms;
  RNArray<PDBBond *> bonds;
  RNScalar conservation;
  RNMark mark;
  RNScalar value;
  void *data;
};



// Inline functions

inline int PDBResidue::
ID(void) const
{
  // Return id
  return id;
}



inline const char *PDBResidue::
Name(void) const
{
  // Return name
  return name;
}



inline int PDBResidue::
Sequence(void) const
{
  // Return sequence
  return sequence;
}



inline int PDBResidue::
InsertionCode(void) const
{
  // Return insertion code
  return insertion_code;
}



inline const RNRgb& PDBResidue::
Color(void) const
{
  // Return bounding box (include radius)
  static const RNRgb default_color(0, 1, 0);
  return (aminoacid) ? aminoacid->Color() : default_color;
}



inline const R3Box& PDBResidue::
BBox(void) const
{
  // Return bounding box (including atom radii)
  return bbox;
}



inline R3Point PDBResidue::
Centroid(void) const
{
  // Return center of mass
  return PDBCentroid(atoms);
}



inline RNLength PDBResidue::
Radius(void) const
{
  // Return max distance to centroid
  return PDBMaxDistance(atoms, Centroid());
}



inline RNBoolean PDBResidue::
IsMarked(void) const
{
  // Return whether residue is marked
  return (mark == PDBmark);
}



inline RNScalar PDBResidue::
Value(void) const
{
  // Return scalar value associated with residue by user (this is not used by the PDB package)
  return value;
}



inline void *PDBResidue::
Data(void) const
{
  // Return data pointer associated with residue by user (this is not used by the PDB package)
  return data;
}



inline int PDBResidue::
NAtoms(void) const
{
  // Return number of atoms
  return atoms.NEntries();
}



inline PDBAtom *PDBResidue::
Atom(int k) const
{
  // Return Kth atom
  return atoms.Kth(k);
}



inline int PDBResidue::
NBonds(void) const
{
  // Return number of bonds
  return bonds.NEntries();
}



inline PDBBond *PDBResidue::
Bond(int k) const
{
  // Return Kth bond
  return bonds.Kth(k);
}



inline PDBAminoAcid *PDBResidue::
AminoAcid(void) const
{
  // Return amino acid
  return aminoacid;
}



inline PDBChain *PDBResidue::
Chain(void) const
{
  // Return chain
  return chain;
}



inline void PDBResidue::
SetName(const char *name)
{
  // Set name
  strncpy(this->name, name, 4);
  this->name[3] = '\0';
}



inline void PDBResidue::
SetData(void *data)
{
  // Set data pointer associated with residue by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBResidue::
SetValue(RNScalar value)
{
  // Set scalar value associated with residue by user (this is not used by the PDB package)
  this->value = value;
}



inline void PDBResidue::
SetMark(void) 
{
  // Mark this residue
  mark = PDBmark;
}



inline void PDBResidue::
UnsetMark(void) 
{
  // Unmark this residue
  mark = 0;
}



inline PDBAtom *PDBResidue:: 
FindAtom(const char *str) const
{
  // Find atom matching string
  PDBAtom *atom = NULL;
  if (FindAny(str, &atom, PDB_ATOM) == PDB_ATOM) return atom;
  else return NULL;
}




