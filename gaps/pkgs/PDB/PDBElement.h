// Include file for PDBElement class



// Element ids

typedef enum {
  PDB_H_ELEMENT,
  PDB_C_ELEMENT,
  PDB_N_ELEMENT,
  PDB_O_ELEMENT,
  PDB_P_ELEMENT,
  PDB_S_ELEMENT,
  PDB_CA_ELEMENT,
  PDB_FE_ELEMENT,
  PDB_ZN_ELEMENT,
  PDB_CD_ELEMENT,
  PDB_I_ELEMENT,  
  PDB_NUM_ELEMENTS
} PDBElementID;



// Class declaration

class PDBElement {
public:
  // Constructor
  PDBElement(int id, const char *name, int atomic_number, RNScalar atomic_weight,
    RNLength covalent_radius, RNLength van_der_waals_radius, RNLength united_atom_radius,
    RNScalar electronegativity, RNScalar charge, const RNRgb& color);

  // Properties
  int ID(void) const;
  const char *Name(void) const;
  int AtomicNumber(void) const;
  RNScalar AtomicWeight(void) const;
  RNLength Radius(void) const;
  RNScalar Electronegativity(void) const;
  RNScalar Charge(void) const;
  const RNRgb& Color(void) const;
  RNBoolean IsMarked(void) const;
  RNScalar Value(void) const;
  void *Data(void) const;

  // Manipulation functions
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

public:
  // Data fields
  int id;
  char name[4];
  int atomic_number;
  RNScalar atomic_weight;
  RNLength covalent_radius;
  RNLength van_der_waals_radius;
  RNLength united_atom_radius;
  RNScalar electronegativity;
  RNScalar charge;
  RNRgb color;
  RNMark mark;
  RNScalar value;
  void *data;
};



// Public variables

extern const int PDBnelements;
extern PDBElement PDBelements[PDB_NUM_ELEMENTS];



// Public functions

PDBElement *PDBFindElement(const char *name);



// Inline functions

inline int PDBElement::
ID(void) const
{
  // Return id
  return id;
}



inline const char *PDBElement::
Name(void) const
{
  // Return name
  return name;
}



inline int PDBElement::
AtomicNumber(void) const
{
  // Return atomic number
  return atomic_number;
}



inline RNScalar PDBElement::
AtomicWeight(void) const
{
  // Return atomic weight
  return atomic_weight;
}



inline RNLength PDBElement::
Radius(void) const
{
  // Return united atom radius
  return united_atom_radius;
}



inline RNScalar PDBElement::
Electronegativity(void) const
{
  // Return electronegativity
  return electronegativity;
}



inline RNScalar PDBElement::
Charge(void) const
{
  // Return charge
  return charge;
}



inline const RNRgb& PDBElement::
Color(void) const
{
  // Return color
  return color;
}



inline RNBoolean PDBElement::
IsMarked(void) const
{
  // Return whether element is marked
  return (mark == PDBmark);
}



inline RNScalar PDBElement::
Value(void) const
{
  // Return scalar value associated with element by user (this is not used by the PDB package)
  return value;
}



inline void *PDBElement::
Data(void) const
{
  // Return data pointer associated with element by user (this is not used by the PDB package)
  return data;
}



inline void PDBElement::
SetData(void *data)
{
  // Set data pointer associated with element by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBElement::
SetValue(RNScalar value)
{
  // Set scalar value associated with element by user (this is not used by the PDB package)
  this->value = value;
}



inline void PDBElement::
SetMark(void) 
{
  // Mark this element
  mark = PDBmark;
}



inline void PDBElement::
UnsetMark(void) 
{
  // Unmark this element
  mark = 0;
}


