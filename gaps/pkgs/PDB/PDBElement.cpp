// Source file for PDBElement



// Include files

#include "PDB.h"




// Radius info from http://www.umass.edu/microbio/rasmol/rasbonds.htm
// Color info from http://www.imb-jena.de/ImgLibDoc/help/color/
// Atomic weight info from http://education.jlab.org/itselemental/index.html
// Charge info computed with CHARMM - Brooks,B.R., Bruccoleri,R.E., Olafson,B.D., States,D.J., Swaminathan,S. and Karplus,M. (1983)
//   as used in GRASP - Nicholls A., Sharp K., and Honig B., PROTEINS, Structure, Function and Genetics, Vol. 11, No.4, 1991, pg. 281

const int PDBnelements = PDB_NUM_ELEMENTS;
PDBElement PDBelements[PDB_NUM_ELEMENTS] = 
{ 
  PDBElement(PDB_H_ELEMENT,   "H",  1,    1.008, 0.320, 1.100, 1.100, 2.10,  PDB_UNKNOWN, RNRgb(0.5, 0.5, 0.5) ), 
  PDBElement(PDB_C_ELEMENT,   "C",  6,   12.011, 0.720, 1.548, 1.872, 2.55,  0.55,        RNRgb(0, 1, 0)       ), 
  PDBElement(PDB_N_ELEMENT,   "N",  7,   14.007, 0.680, 1.400, 1.507, 3.04, -0.10,        RNRgb(0, 0, 1)       ), 
  PDBElement(PDB_O_ELEMENT,   "O",  8,   15.999, 0.680, 1.348, 1.400, 3.44, -0.55,        RNRgb(1, 0, 0)       ), 
  PDBElement(PDB_P_ELEMENT,   "P", 15,   30.974, 1.036, 1.880, 1.880, 2.19,  0.75,        RNRgb(1, 0, 1)       ), // Charge is guess
  PDBElement(PDB_S_ELEMENT,   "S", 16,   32.065, 1.020, 1.808, 1.848, 2.58,  0.55,        RNRgb(1, 1, 0)       ), // Charge is guess
  PDBElement(PDB_CA_ELEMENT, "CA", 20,   40.078, 0.992, 1.948, 1.948, 1.00,  0.10,        RNRgb(0, 1, 1)       ),
  PDBElement(PDB_FE_ELEMENT, "FE", 26,   55.845, 1.420, 1.948, 1.948, 1.83,  0.00,        RNRgb(1, 0.5, 0.5)   ),
  PDBElement(PDB_ZN_ELEMENT, "ZN", 30,   65.409, 1.448, 1.148, 1.148, 1.65,  2.00,        RNRgb(0.5, 1, 0.5)   ),
  PDBElement(PDB_CD_ELEMENT, "CD", 48,  112.411, 1.688, 1.748, 1.748, 1.69,  PDB_UNKNOWN, RNRgb(0.5, 0.5, 1)   ),
  PDBElement(PDB_I_ELEMENT,   "I", 53,  126.904, 1.400, 1.748, 1.748, 2.66,  PDB_UNKNOWN, RNRgb(0.5, 0.3, 0.1) ),
};




PDBElement::
PDBElement(int id, const char *name, int atomic_number, RNScalar atomic_weight,
  RNLength covalent_radius, RNLength van_der_waals_radius, RNLength united_atom_radius, 
  RNScalar electronegativity, RNScalar charge, const RNRgb& color)
  : id(id),
    atomic_number(atomic_number),
    atomic_weight(atomic_weight),
    covalent_radius(covalent_radius), 
    van_der_waals_radius(van_der_waals_radius), 
    united_atom_radius(united_atom_radius),
    electronegativity(electronegativity),
    charge(charge),
    color(color),
    mark(0),
    value(0),
    data(NULL)
{
  // Copy name
  if (name) { strncpy(this->name, name, 4); this->name[3] = 0; }
  else this->name[0] = '\0';
}



PDBElement *
PDBFindElement(const char *name)
{
  // Search for element with matching name
  for (int i = 0; i < PDBnelements; i++) {
    PDBElement *element = &PDBelements[i];
    if (!strcmp(name, element->name)) {
      return element;
    }
  }

  // Return "?" element
  // printf("Unknown element: %s\n", name);
  return NULL;
}



