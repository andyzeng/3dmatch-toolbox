// Include file for PDBAminoAcid class



// Amino acid IDs

typedef enum {
  PDB_ALA_AMINOACID,
  PDB_CYS_AMINOACID,
  PDB_ASP_AMINOACID,
  PDB_GLU_AMINOACID,
  PDB_PHE_AMINOACID,
  PDB_GLY_AMINOACID,
  PDB_HIS_AMINOACID,
  PDB_ILE_AMINOACID,
  PDB_LYS_AMINOACID,
  PDB_LEU_AMINOACID,
  PDB_MET_AMINOACID,
  PDB_ASN_AMINOACID,
  PDB_PRO_AMINOACID,
  PDB_GLN_AMINOACID,
  PDB_ARG_AMINOACID,
  PDB_SER_AMINOACID,
  PDB_THR_AMINOACID,
  PDB_VAL_AMINOACID,
  PDB_TRP_AMINOACID,
  PDB_TYR_AMINOACID,
  PDB_NUM_AMINOACIDS
} PDBAminoAcidID;



// Amino acid groups
// From http://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html

typedef enum {
  PDB_ALIPHATIC_AMINOACID_GROUP,
  PDB_AROMATIC_AMINOACID_GROUP,
  PDB_ACIDIC_AMINOACID_GROUP,
  PDB_BASIC_AMINOACID_GROUP,
  PDB_HYDROXYLIC_AMINOACID_GROUP,
  PDB_SULFUR_AMINOACID_GROUP,
  PDB_AMIDIC_AMINOACID_GROUP,
  PDB_NUM_AMINOACID_GROUPS
} PDBAminoAcidGroup;



// Class declaration

class PDBAminoAcid {
public:
  // Constructor
  PDBAminoAcid(int id, const char *name, const char *code, char letter, int group, 
    int natoms, RNScalar hydrophobicity, const RNRgb& color);

  // Properties
  int ID(void) const;
  const char *Name(void) const;
  const char *Code(void) const;
  char Letter(void) const;
  int Group(void) const;
  RNScalar Hydrophobicity(void) const;
  const RNRgb& Color(void) const;
  RNBoolean IsMarked(void) const;
  RNScalar Value(void) const;
  void *Data(void) const;

  // Properties for each atom of amino acid
  int NAtoms(void) const;
  const char *AtomName(int k) const;
  PDBElement *AtomElement(int k) const;
  RNScalar AtomCharge(int k) const;

  // Manipulation functions
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

public:
  // Data fields
  int id;
  char name[16];
  char code[4];
  char letter;
  int group;
  int natoms;
  RNScalar hydrophobicity;
  RNRgb color;
  RNMark mark;
  RNScalar value;
  void *data;
};



// Public variables

extern const int PDBnaminoacids;
extern PDBAminoAcid PDBaminoacids[PDB_NUM_AMINOACIDS];



// Public functions

PDBAminoAcid *PDBFindAminoAcid(const char *name);



// Inline functions

inline int PDBAminoAcid::
ID(void) const
{
  // Return id
  return id;
}



inline const char *PDBAminoAcid::
Name(void) const
{
  // Return name
  return name;
}



inline const char *PDBAminoAcid::
Code(void) const
{
  // Return three-letter code representing name
  return code;
}



inline char PDBAminoAcid::
Letter(void) const
{
  // Return letter symbol
  return letter;
}



inline int PDBAminoAcid::
NAtoms(void) const
{
  // Return number of atoms
  return natoms;
}



inline int PDBAminoAcid::
Group(void) const
{
  // Return group
  return group;
}



inline RNScalar PDBAminoAcid::
Hydrophobicity(void) const
{
  // Return hydrophobicity
  return hydrophobicity;
}



inline const RNRgb& PDBAminoAcid::
Color(void) const
{
  // Return color
  return color;
}



inline RNBoolean PDBAminoAcid::
IsMarked(void) const
{
  // Return whether amino acid is marked
  return (mark == PDBmark);
}



inline RNScalar PDBAminoAcid::
Value(void) const
{
  // Return scalar value associated with amino acid by user (this is not used by the PDB package)
  return value;
}



inline void *PDBAminoAcid::
Data(void) const
{
  // Return data pointer associated with amino acid by user (this is not used by the PDB package)
  return data;
}



inline void PDBAminoAcid::
SetData(void *data)
{
  // Set data pointer associated with amino acid by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBAminoAcid::
SetValue(RNScalar value)
{
  // Set scalar value associated with amino acid by user (this is not used by the PDB package)
  this->value = value;
}



inline void PDBAminoAcid::
SetMark(void) 
{
  // Mark this amino acid
  mark = PDBmark;
}



inline void PDBAminoAcid::
UnsetMark(void) 
{
  // Unmark this amino acid
  mark = 0;
}




