// Include file for PDBBond class



// Bond types 

typedef enum {
  PDB_UNKNOWN_BOND,
  PDB_HYDROGEN_BOND,
  PDB_NONCOVALENT_BOND,
  PDB_COVALENT_BOND,
  PDB_NUM_BOND_TYPES
} PDBBondType;



// Class declaration

class PDBBond {
public:
  // Constructor
  PDBBond(PDBAtom *atom1, PDBAtom *atom2, PDBBondType type = PDB_UNKNOWN_BOND);
  ~PDBBond(void);

  // Properties
  int ID(void) const;
  PDBBondType Type(void) const;
  RNLength Length(void) const;
  const RNRgb& Color(void) const;
  RNBoolean IsMarked(void) const;
  RNScalar Value(void) const;
  void *Data(void) const;

  // Atom access functions
  int NAtoms(void) const;
  PDBAtom *Atom(int k) const;
  PDBAtom *OtherAtom(PDBAtom *atom) const;

  // Manipulation functions
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

public:
  // Data fields
  PDBAtom *atoms[2];
  PDBBondType type;
  RNMark mark;
  RNScalar value;
  void *data;
  int id;
};



// Inline functions

inline int PDBBond::
ID(void) const
{
  // Return id
  return id;
}



inline PDBBondType PDBBond::
Type(void) const
{
  // Return bond type
  return type;
}



inline RNLength PDBBond::
Length(void) const
{
  // Return bond length
  return R3Distance(atoms[0]->Position(), atoms[1]->Position());
}



inline RNBoolean PDBBond::
IsMarked(void) const
{
  // Return whether bond is marked
  return (mark == PDBmark);
}



inline RNScalar PDBBond::
Value(void) const
{
  // Return scalar value associated with bond by user (this is not used by the PDB package)
  return value;
}



inline void *PDBBond::
Data(void) const
{
  // Return data pointer associated with bond by user (this is not used by the PDB package)
  return data;
}



inline int PDBBond::
NAtoms(void) const
{
  // Return number of atoms in bond
  return 2;
}



inline PDBAtom *PDBBond::
Atom(int k) const
{
  // Return kth atom 
  assert((k == 0) || (k == 1));
  return atoms[k];
}



inline PDBAtom *PDBBond::
OtherAtom(PDBAtom *atom) const
{
  // Return other atom 
  assert((atom == atoms[0]) || (atom == atoms[1]));
  return (atom == atoms[0]) ? atoms[1] : atoms[0];
}



inline void PDBBond::
SetData(void *data)
{
  // Set data pointer associated with bond by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBBond::
SetValue(RNScalar value)
{
  // Set scalar value associated with bond by user (this is not used by the PDB package)
  this->value = value;
}



inline void PDBBond::
SetMark(void) 
{
  // Mark this bond
  mark = PDBmark;
}



inline void PDBBond::
UnsetMark(void) 
{
  // Unmark this bond
  mark = 0;
}


