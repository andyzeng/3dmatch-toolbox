// Include file for PDBAtom class



// Class declaration

class PDBAtom {
public:
  // Constructor
  PDBAtom(PDBModel *model, PDBChain *chain, PDBResidue *residue, PDBElement *element,
          int serial, const char *atom_name, int altLoc, RNScalar x, RNScalar y, RNScalar z, 
          RNScalar occupancy, RNScalar tempFactor, RNScalar charge, RNBoolean hetatm = FALSE);
  ~PDBAtom(void);

  // Properties
  int ID(void) const;
  int Serial(void) const;
  const char *Name(void) const;
  const R3Point& Position(void) const;
  RNScalar Occupancy(void) const;
  RNScalar TempFactor(void) const;
  RNScalar Charge(void) const;
  int AlternateLocation(void) const;
  RNScalar Radius(void) const;
  RNScalar Hydrophobicity(void) const;
  const RNRgb& Color(void) const;
  R3Box BBox(void) const;
  RNBoolean IsHetAtom(void) const;
  RNBoolean IsBackbone(void) const;
  RNBoolean IsBonded(PDBAtom *atom);
  RNBoolean IsBonded(PDBResidue *residue);
  RNBoolean IsMarked(void) const;
  RNScalar Value(void) const;
  void *Data(void) const;

  // Bond access functions
  int NBonds(void) const;
  PDBBond *Bond(int k) const;

  // Other access functions
  PDBElement *Element(void) const;
  PDBResidue *Residue(void) const;
  PDBAminoAcid *AminoAcid(void) const;
  PDBChain *Chain(void) const;
  PDBModel *Model(void) const;
  PDBFile *File(void) const;

  // Manipulation functions
  void Transform(const R3Affine& affine);
  void SetPosition(const R3Point& position);
  void SetName(const char *name);
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

public:
  // Straight from PDB file
  int serial;
  char name[8];
  int altLoc;
  R3Point position;
  RNScalar occupancy;
  RNScalar tempFactor;
  RNScalar charge;
  RNBoolean hetatm;

public:
  // Other data fields
  int id;
  PDBResidue *residue;
  PDBElement *element;
  RNArray<PDBBond *> bonds;
  RNScalar accessible_surface_area;
  int aminoacid_atom_type;
  RNMark mark;

public:
  // User-definable data fields
  RNScalar value;
  void *data;
};



// Inline functions


inline int PDBAtom::
ID(void) const
{
  // Return id
  return id;
}



inline int PDBAtom::
Serial(void) const
{
  // Return serial number
  return serial;
}



inline const char *PDBAtom::
Name(void) const
{
  // Return name
  return name;
}



inline const R3Point& PDBAtom::
Position(void) const
{
  // Return position
  return position;
}



inline RNScalar PDBAtom::
Occupancy(void) const
{
  // Return occupancy
  return occupancy;
}



inline RNScalar PDBAtom::
TempFactor(void) const
{
  // Return temperature factor
  return tempFactor;
}



inline RNScalar PDBAtom::
Charge(void) const
{
  // Return charge
  return charge;
}



inline int PDBAtom::
AlternateLocation(void) const
{
  // Return altLoc code
  return altLoc;
}



inline RNLength PDBAtom::
Radius(void) const
{
  // Return bounding box (include radius)
  return (element) ? element->Radius() : 1.8;
}



inline RNScalar PDBAtom::
Hydrophobicity(void) const
{
  // Return hydrophobicity
  PDBAminoAcid *aminoacid = AminoAcid();
  return (aminoacid) ? aminoacid->Hydrophobicity() : PDB_UNKNOWN;
}



inline const RNRgb& PDBAtom::
Color(void) const
{
  // Return bounding box (include radius)
  static const RNRgb default_color(0.7, 0.7, 0.7);
  return (element) ? element->Color() : default_color;
}



inline R3Box PDBAtom::
BBox(void) const
{
  // Return bounding box (include radius)
  return R3Box(Position() - Radius() * R3ones_vector, Position() + Radius() * R3ones_vector);
}



inline RNBoolean PDBAtom::
IsHetAtom(void) const
{
  // Return whether atom is a HETATM
  return hetatm;
}



inline RNBoolean PDBAtom::
IsBackbone(void) const
{
  // Return whether atom is on backbone of protein
  if (aminoacid_atom_type == PDB_NULL_ATOM) return FALSE;
  return ((PDBaminoacid_atoms[aminoacid_atom_type].group == PDB_C_ATOM_GROUP) ||
          (PDBaminoacid_atoms[aminoacid_atom_type].group == PDB_CH1E_ATOM_GROUP) ||
          (PDBaminoacid_atoms[aminoacid_atom_type].group == PDB_NH1_ATOM_GROUP) ||
          (PDBaminoacid_atoms[aminoacid_atom_type].group == PDB_CH2G_ATOM_GROUP));
}



inline RNBoolean PDBAtom::
IsMarked(void) const
{
  // Return whether atom is marked
  return (mark == PDBmark);
}



inline RNScalar PDBAtom::
Value(void) const
{
  // Return scalar value associated with atom by user (this is not used by the PDB package)
  return value;
}



inline void *PDBAtom::
Data(void) const
{
  // Return data pointer associated with atom by user (this is not used by the PDB package)
  return data;
}



inline int PDBAtom::
NBonds(void) const
{
  // Return number of bonds
  return bonds.NEntries();
}



inline PDBBond *PDBAtom::
Bond(int k) const
{
  // Return Kth bond
  return bonds.Kth(k);
}



inline PDBElement *PDBAtom::
Element(void) const
{
  // Return element
  return element;
}



inline PDBResidue *PDBAtom::
Residue(void) const
{
  // Return residue
  return residue;
}



inline void PDBAtom::
SetName(const char *name)
{
  // Set name
  strncpy(this->name, name, 8);
  this->name[7] = '\0';
}



inline void PDBAtom::
SetData(void *data)
{
  // Set data pointer associated with atom by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBAtom::
SetValue(RNScalar value)
{
  // Set scalar value associated with atom by user (this is not used by the PDB package)
  this->value = value;;
}



inline void PDBAtom::
SetMark(void) 
{
  // Mark this atom
  mark = PDBmark;
}



inline void PDBAtom::
UnsetMark(void) 
{
  // Unmark this atom
  mark = 0;
}


