// Source file for PDBResidue



// Include files

#include "PDB.h"



// Turn off warning about lack of const in atom_names (fix this)
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wwrite-strings"
#endif



PDBResidue::
PDBResidue(PDBModel *model, PDBChain *chain, const char *name, int sequence, int insertion_code)
  : sequence(sequence),
    insertion_code(insertion_code),
    bbox(R3null_box),
    chain(chain), 
    aminoacid(NULL),
    conservation(PDB_UNKNOWN),
    mark(0),
    value(0),
    data(NULL)
{
  // Just checking
  assert(!model || !chain || (chain->model == model));

  // Assign ID
  static int PDBnext_residue_id = 0;
  id = PDBnext_residue_id++;

  // Copy name
  if (name) { strncpy(this->name, name, 4); this->name[3] = 0; }
  else this->name[0] = '\0';

  // Find amino acid
  if (name[0]) aminoacid = PDBFindAminoAcid(name);

  // Insert residue into chain and model
  if (chain) chain->residues.Insert(this);
  if (model) model->residues.Insert(this);
}




PDBResidue::
~PDBResidue(void)
{
  // Delete atoms
  while (NAtoms()) delete atoms.Tail();

  // Remove residue from chain, and model
  PDBChain *chain = Chain();
  if (chain) {
    chain->residues.Remove(this);
    PDBModel *model = chain->Model();
    if (model) {
      model->residues.Remove(this);
    }
  }
}



PDBModel *PDBResidue::
Model(void) const
{
  // Return model
  return (chain) ? chain->Model() : NULL;
}



PDBFile *PDBResidue::
File(void) const
{
  // Return file
  return (chain) ? chain->File() : NULL;
}



RNBoolean PDBResidue::
HasHetAtoms(void) const
{
  // Check whether residue has any hetatoms
  for (int i = 0; i < NAtoms(); i++) 
    if (Atom(i)->IsHetAtom()) return TRUE;
  return FALSE;
}



RNBoolean PDBResidue::
IsBonded(PDBAtom *atom2)
{
  // Search for bond to atom2 from any atom in this residue 
  for (int i = 0; i < NBonds(); i++) {
    PDBBond *bond = Bond(i);
    if (bond->Atom(0) == atom2) return TRUE;
    if (bond->Atom(1) == atom2) return TRUE;
  }

  // Bond not found
  return FALSE;
}



RNBoolean PDBResidue::
IsBonded(PDBResidue *residue2)
{
  // Search for bond to any atom in residue2 from any atom in this residue
  for (int i = 0; i < NBonds(); i++) {
    PDBBond *bond = Bond(i);
    if (bond->Atom(0)->Residue() == residue2) return TRUE;
    if (bond->Atom(1)->Residue() == residue2) return TRUE;
  }

  // Bond not found
  return FALSE;
}



void PDBResidue::
Transform(const R3Affine& affine)
{
  // Transform all atoms by affine transformation
  bbox = R3null_box;
  for (int i = 0; i < atoms.NEntries(); i++) 
    atoms[i]->Transform(affine);
}



PDBAtom *PDBResidue::
InsertCopy(PDBAtom *a)
{
  // Insert copy of atom into this residue
  PDBAtom *atom = new PDBAtom(Model(), Chain(), this, 
    a->Element(), a->Serial(), a->Name(), a->AlternateLocation(), 
    a->Position().X(), a->Position().Y(), a->Position().Z(), 
    a->Occupancy(), a->TempFactor(), a->Charge(), a->IsHetAtom());
  assert(atom);

  // return atom
  return atom;
}



PDBAtom *PDBResidue::
FindAtom(const char *atom_name, int atom_serial, int altLoc) const
{
  // Check all atoms
  for (int i = 0; i < NAtoms(); i++) {
    PDBAtom *atom = Atom(i);
    if ((atom_serial == atom->Serial()) &&
        ((altLoc == 0) || (altLoc == atom->AlternateLocation())) &&
        (!strcmp(atom_name, atom->Name()))) {
      return atom;
    }
  }
  return NULL;
}



int PDBResidue::
FindAtoms(char **atom_names, PDBAtom **atom_ptrs, int natoms)
{
  // Find atoms by name
  int atom_count = 0;
  for (int i = 0; i < natoms; i++) {
    atom_ptrs[i] = NULL;
    for (int j = 0; j < NAtoms(); j++) {
      PDBAtom *atom = Atom(j);
      if (atom->IsHetAtom()) continue;
      if (!strcmp(atom->Name(), atom_names[i])) {
        atom_ptrs[i] = atom;
        atom_count++;
        break;
      }
    }
  }

  // Return number of atoms found
  return atom_count;
}



PDBStructureType PDBResidue::
FindAny(const char *str, PDBAtom **a, PDBStructureType maxlevel) const
{
  // Check string
  if (!str) return PDB_RESIDUE;
  if (strlen(str) < 1) return PDB_RESIDUE;

  // Parse token
  char *strp = (char *) str;
  while (*strp) { if (*strp == '.') break; strp++; }

  // Temporarily modify string
  char save = *strp;
  *strp = '\0';

  // Find atom from string
  PDBAtom *result = NULL;
  for (int i = 0; i < NAtoms(); i++) {
    PDBAtom *atom = Atom(i);
    if (!strcmp(atom->Name(), str)) {
      result = atom;
      break;
    }
  }

  // Restore string
  *strp = save;

  // Fill return values and consider matches at deeper levels
  PDBStructureType level = PDB_RESIDUE;
  if (result) {
    level = PDB_ATOM;
    if (a) *a = result;
  }

  // Return level of match 
  return level;
}



int PDBResidue::
ConstructCoordinateSystem(R3CoordSystem& cs, int coordinate_system_type)
{
  // Initialize origin and axes
  R3Point origin = R3zero_point;
  R3Vector posx = R3zero_vector;
  R3Vector posy = R3zero_vector;
  R3Vector posz = R3zero_vector;

  // Check coordinate system type 
  if (coordinate_system_type == 0) {
    // Make coordinate system with backbone atoms
    const int natoms = 3;
    PDBAtom *atom_ptrs[natoms];
    char *atom_names[natoms] = { " CA ", " C  ", " N  " };
    if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
    origin = atom_ptrs[0]->Position();
    R3Vector c_vector = atom_ptrs[1]->Position() - origin;
    R3Vector n_vector = atom_ptrs[2]->Position() - origin;
    posz = c_vector;
    posx = n_vector;
  }
  else if (coordinate_system_type == 1) {
    // Get amino acid
    PDBAminoAcid *aminoacid = AminoAcid();
    if (!aminoacid) return 0;

    // Construct coordinate frame based on side chain atoms as in Singh and Thornton
    switch (aminoacid->Letter()) {
    case 'A': { // ALA
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CA ", " CB ", " N  " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector cb_vector = atom_ptrs[1]->Position() - origin;
      R3Vector n_vector = atom_ptrs[2]->Position() - origin;
      posx = -cb_vector;
      posz = n_vector % cb_vector;
    } break;

    case 'C': { // CYS
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CB ", " SG ", " CA " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector sg_vector = atom_ptrs[1]->Position() - origin;
      R3Vector ca_vector = atom_ptrs[2]->Position() - origin;
      posx = -sg_vector;
      posz = ca_vector % sg_vector;
    } break;

    case 'D': { // ASP
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CG ", " OD1", " OD2" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector od1_vector = atom_ptrs[1]->Position() - origin;
      R3Vector od2_vector = atom_ptrs[2]->Position() - origin;
      posx = -(od1_vector + od2_vector);
      posz = od1_vector % od2_vector;
    } break;

    case 'E': { // GLU
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CD ", " OE1", " OE2" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector oe1_vector = atom_ptrs[1]->Position() - origin;
      R3Vector oe2_vector = atom_ptrs[2]->Position() - origin;
      posx = -(oe1_vector + oe2_vector);
      posz = oe1_vector % oe2_vector;
    } break;

    case 'F': { // PHE
      const int natoms = 7;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CG ", " CD2", " CE2", " CZ ", " CE1", " CD1", " CB " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = R3zero_point;
      RNArray<R3Point *> points;
      for (int i = 0; i < natoms; i++) {
        points.Insert(&(atom_ptrs[i]->position));
        if (i < 6) origin += atom_ptrs[i]->Position();
      }
      R3Plane plane(points);
      origin /= 6;
      posx = atom_ptrs[0]->Position() - origin;
      posz = plane.Normal();
    } break;

    case 'G': { // GLY
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CA ", " C  ", " N  " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector c_vector = atom_ptrs[1]->Position() - origin;
      R3Vector n_vector = atom_ptrs[2]->Position() - origin;
      posx = c_vector + n_vector;
      posz = n_vector % c_vector;
    } break;

    case 'H': { // HIS
      const int natoms = 5;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CG ", " CD2", " NE2", " CE1", " ND1" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = R3zero_point;
      RNArray<R3Point *> points;
      for (int i = 0; i < 5; i++) {
        points.Insert(&(atom_ptrs[i]->position));
        origin += atom_ptrs[i]->Position();
      }
      R3Plane plane(points);
      origin /= natoms;
      posy = atom_ptrs[1]->Position() - origin;
      posz = plane.Normal();
      posx = posy % posz;
    } break;

    case 'I': { // ILE
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CB ", " CG2", " CG1" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector cg2_vector = atom_ptrs[1]->Position() - origin;
      R3Vector cg1_vector = atom_ptrs[2]->Position() - origin;
      posx = -(cg2_vector + cg1_vector);
      posz = cg2_vector % cg1_vector;
    } break;

    case 'K': { // LYS
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CE ", " NZ ", " CD " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector nz_vector = atom_ptrs[1]->Position() - origin;
      R3Vector cd_vector = atom_ptrs[2]->Position() - origin;
      posx = -nz_vector;
      posz = cd_vector % nz_vector;
    } break;

    case 'L': { // LEU
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CG ", " CD2", " CD1" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector cd2_vector = atom_ptrs[1]->Position() - origin;
      R3Vector cd1_vector = atom_ptrs[2]->Position() - origin;
      posx = -(cd2_vector + cd1_vector);
      posz = cd2_vector % cd1_vector;
    } break;

    case 'M': { // MET
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " SD ", " CG ", " CE " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector cg_vector = atom_ptrs[1]->Position() - origin;
      R3Vector ce_vector = atom_ptrs[2]->Position() - origin;
      posx = cg_vector + ce_vector;
      posz = cg_vector % ce_vector;
    } break; 

    case 'N': { // ASN
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CG ", " OD1", " ND2" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector od1_vector = atom_ptrs[1]->Position() - origin;
      R3Vector nd2_vector = atom_ptrs[2]->Position() - origin;
      posx = -(od1_vector + nd2_vector);
      posz = od1_vector % nd2_vector;
    } break;

    case 'P': { // PRO
      const int natoms = 5;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CA ", " N  ", " CD ", " CG ", " CB " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      RNArray<R3Point *> points;
      for (int i = 0; i < 5; i++) points.Insert(&(atom_ptrs[i]->position));
      R3Plane plane(points);
      posx = atom_ptrs[1]->Position() - origin;
      posz = plane.Normal();
    } break;

    case 'Q': { // GLN
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CD ", " OE1", " NE2" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector oe1_vector = atom_ptrs[1]->Position() - origin;
      R3Vector ne2_vector = atom_ptrs[2]->Position() - origin;
      posx = -(oe1_vector + ne2_vector);
      posz = oe1_vector % ne2_vector;
    } break;

    case 'R': { // ARG
      const int natoms = 5;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CZ ", " NE ", " CD ", " NH1", " NH2" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      RNArray<R3Point *> points;
      for (int i = 0; i < 5; i++) points.Insert(&(atom_ptrs[i]->position));
      R3Plane plane(points);
      posx = atom_ptrs[1]->Position() - origin;
      posz = plane.Normal();
    } break;

    case 'S': { // SER
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CB ", " OG ", " CA " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector og_vector = atom_ptrs[1]->Position() - origin;
      R3Vector ca_vector = atom_ptrs[2]->Position() - origin;
      posx = -og_vector;
      posz = ca_vector % og_vector;
    } break;

    case 'T': { // THR
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CB ", " OG1", " CG2" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector og1_vector = atom_ptrs[1]->Position() - origin;
      R3Vector cg2_vector = atom_ptrs[2]->Position() - origin;
      posx = -(og1_vector + cg2_vector);
      posz = og1_vector % cg2_vector;
    } break;

    case 'V': { // VAL
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CB ", " CG2", " CG1" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector cg2_vector = atom_ptrs[1]->Position() - origin;
      R3Vector cg1_vector = atom_ptrs[2]->Position() - origin;
      posx = -(cg2_vector + cg1_vector);
      posz = cg2_vector % cg1_vector;
    } break;

    case 'W': { // TRP
      const int natoms = 9;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CH2", " CZ3", " CE3", " CD2", " CG ", " CD1", " NE1", " CE2", " CZ2" };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      RNArray<R3Point *> points;
      for (int i = 0; i < natoms; i++) points.Insert(&(atom_ptrs[i]->position));
      R3Plane plane(points);
      origin = 0.5 * (atom_ptrs[3]->Position() + atom_ptrs[7]->Position());
      posx = atom_ptrs[3]->Position() - origin;
      posz = plane.Normal();
    } break;

    case 'Y': { // TYR
      const int natoms = 8;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CG ", " CD2", " CE2", " CZ ", " OH ", " CE1", " CD1", " CB " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = R3zero_point;
      RNArray<R3Point *> points;
      for (int i = 0; i < natoms; i++) {
        points.Insert(&(atom_ptrs[i]->position));
        if ((i != 4) && (i != 7)) origin += atom_ptrs[i]->Position();
      }
      R3Plane plane(points);
      origin /= 6;
      posx = atom_ptrs[0]->Position() - origin;
      posz = plane.Normal();
    } break;

    default:
      fprintf(stderr, "Unknown amino acid type: %c\n", aminoacid->Letter());
      return 0;
    }
  }
  else if (coordinate_system_type == 2) {
    // Get amino acid
    PDBAminoAcid *aminoacid = AminoAcid();
    if (!aminoacid) return 0;

    // Make coordinate system with CA and statistics of sidechain atoms
    // Special cases for amino acids with small sidechains
    switch (aminoacid->Letter()) { 
    case 'A': { // ALA
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CA ", " CB ", " N  " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector cb_vector = atom_ptrs[1]->Position() - origin;
      R3Vector n_vector = atom_ptrs[2]->Position() - origin;
      posx = -cb_vector;
      posz = n_vector % cb_vector;
    } break;

    case 'C': { // CYS
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CB ", " SG ", " CA " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector sg_vector = atom_ptrs[1]->Position() - origin;
      R3Vector ca_vector = atom_ptrs[2]->Position() - origin;
      posx = -sg_vector;
      posz = ca_vector % sg_vector;
    } break;

    case 'G': { // GLY
      const int natoms = 3;
      PDBAtom *atom_ptrs[natoms];
      char *atom_names[natoms] = { " CA ", " C  ", " N  " };
      if (FindAtoms(atom_names, atom_ptrs, natoms) != natoms) return 0;
      origin = atom_ptrs[0]->Position();
      R3Vector c_vector = atom_ptrs[1]->Position() - origin;
      R3Vector n_vector = atom_ptrs[2]->Position() - origin;
      posx = c_vector + n_vector;
      posz = n_vector % c_vector;
    }  break;

    default: {
      // Get CA and sidechain atoms
      PDBAtom *ca_atom = NULL;
      RNArray<PDBAtom *> sidechain_atoms;
      for (int i = 0; i < NAtoms(); i++) {
        PDBAtom *atom = Atom(i);
        if (!strcmp(atom->Name(), " CA ")) ca_atom = atom;
        else if (strcmp(atom->Name(), " C  ") && strcmp(atom->Name(), " N  ") && strcmp(atom->Name(), " O  ")) {
          sidechain_atoms.Insert(atom);
        }
      }

      // Check atoms
      if (!ca_atom) return 0;
      if (sidechain_atoms.NEntries() < 3) return 0;

      // Compute sidechain centroid
      R3Point sidechain_centroid = R3zero_point;
      for (int i = 0; i < sidechain_atoms.NEntries(); i++) sidechain_centroid += sidechain_atoms[i]->Position();
      sidechain_centroid /= sidechain_atoms.NEntries();

      // Get vector from CA to sidechain centroid
      R3Vector sidechain_vector = sidechain_centroid - ca_atom->Position();
      RNLength sidechain_vector_length = sidechain_vector.Length();
      if (RNIsZero(sidechain_vector_length)) return 0;
      sidechain_vector /= sidechain_vector_length;

      // Get vector of least variation in sidechain
      RNBoolean found = FALSE;
      R3Vector principle_vector;
      R3Triad axes = PDBPrincipleAxes(sidechain_atoms, sidechain_centroid);
      for (int i = 2; i >= 0; i++) {
        R3Vector principle_vector = axes[i];
        RNLength principle_vector_length = principle_vector.Length();
        if (RNIsZero(principle_vector_length)) continue;
        principle_vector /= principle_vector_length;
        if (RNIsEqual(fabs(principle_vector.Dot(sidechain_vector)), 1.0)) continue;
        found = TRUE;
        break;
      }
      if (!found) return 0;

      // Compute coordinate system vectors
      origin = ca_atom->Position();
      posz = sidechain_vector;
      posx = principle_vector;
    } break; }
  }

  // Normalize axes
  RNLength lenx = posx.Length();
  if (RNIsZero(lenx)) return 0;
  else posx /= lenx;
  RNLength lenz = posz.Length();
  if (RNIsZero(lenz)) return 0;
  else posz /= lenz;

  // Orthogonalize axes
  posy = posz % posx;
  posx = posy % posz;

  // Normalize axes again
  lenx = posx.Length();
  if (RNIsZero(lenx)) return 0;
  else posx /= lenx;
  RNLength leny = posy.Length();
  if (RNIsZero(leny)) return 0;
  else posy /= leny;

  // Construct coordinate system from origin and axes
  R3Triad triad(posx, posy, posz);
  cs.SetAxes(triad);
  cs.SetOrigin(origin);

  // Return success
  return 1;
}
