// Source file for PDBAminoAcid class



// Include files

#include "PDB.h"



// Amino acids
// Hydrophobicity data from Young et al., Protein Sci 3:717-729 (1994) 
// Color data from SETOR: hardware-lighted three-dimensional solid model representations of macromolecules. J Mol Graph. 1993, 11, 134-138. 
// Group data from http://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html

const int PDBnaminoacids = PDB_NUM_AMINOACIDS;
PDBAminoAcid PDBaminoacids[PDB_NUM_AMINOACIDS] = 
{
  PDBAminoAcid( PDB_ALA_AMINOACID, "Alanine",       "ALA", 'A', PDB_ALIPHATIC_AMINOACID_GROUP,  10,  0.31, RNRgb(1, 0.5, 0.4)   ),
  PDBAminoAcid( PDB_CYS_AMINOACID, "Cysteine",      "CYS", 'C', PDB_SULFUR_AMINOACID_GROUP,     11,  1.54, RNRgb(1, 1, 0)       ),
  PDBAminoAcid( PDB_ASP_AMINOACID, "Aspartic acid", "ASP", 'D', PDB_ACIDIC_AMINOACID_GROUP,     12, -0.77, RNRgb(1, 0, 0)       ),
  PDBAminoAcid( PDB_GLU_AMINOACID, "Glutamic acid", "GLU", 'E', PDB_ACIDIC_AMINOACID_GROUP,     15, -0.64, RNRgb(1, 0, 0)       ),
  PDBAminoAcid( PDB_PHE_AMINOACID, "Phenylalanine", "PHE", 'F', PDB_AROMATIC_AMINOACID_GROUP,   20,  1.79, RNRgb(1, 0, 1)       ),
  PDBAminoAcid( PDB_GLY_AMINOACID, "Glycine",       "GLY", 'G', PDB_ALIPHATIC_AMINOACID_GROUP,   7,  0.00, RNRgb(0.8, 0.8, 0.8) ),
  PDBAminoAcid( PDB_HIS_AMINOACID, "Histidine",     "HIS", 'H', PDB_BASIC_AMINOACID_GROUP,      17,  0.13, RNRgb(0, 0, 1)       ),
  PDBAminoAcid( PDB_ILE_AMINOACID, "Isoleucine",    "ILE", 'I', PDB_ALIPHATIC_AMINOACID_GROUP,  19,  1.80, RNRgb(1, 0.5, 0.4)   ),
  PDBAminoAcid( PDB_LYS_AMINOACID, "Lysine",        "LYS", 'K', PDB_BASIC_AMINOACID_GROUP,      22, -0.99, RNRgb(0, 0, 1)       ),
  PDBAminoAcid( PDB_LEU_AMINOACID, "Leucine",       "LEU", 'L', PDB_ALIPHATIC_AMINOACID_GROUP,  19,  1.70, RNRgb(1, 0.5, 0.4)   ),
  PDBAminoAcid( PDB_MET_AMINOACID, "Methionine",    "MET", 'M', PDB_SULFUR_AMINOACID_GROUP,     17,  1.23, RNRgb(1, 1, 0)       ),
  PDBAminoAcid( PDB_ASN_AMINOACID, "Asparagine",    "ASN", 'N', PDB_AMIDIC_AMINOACID_GROUP,     14, -0.60, RNRgb(0, 1, 1)       ),
  PDBAminoAcid( PDB_PRO_AMINOACID, "Proline",       "PRO", 'P', PDB_ALIPHATIC_AMINOACID_GROUP,  14,  0.72, RNRgb(1, 0.5, 0.5)   ),
  PDBAminoAcid( PDB_GLN_AMINOACID, "Glutamine",     "GLN", 'Q', PDB_AMIDIC_AMINOACID_GROUP,     17, -0.22, RNRgb(0, 1, 1)       ),
  PDBAminoAcid( PDB_ARG_AMINOACID, "Arginine",      "ARG", 'R', PDB_BASIC_AMINOACID_GROUP,      24, -1.01, RNRgb(0, 0, 1)       ),
  PDBAminoAcid( PDB_SER_AMINOACID, "Serine",        "SER", 'S', PDB_HYDROXYLIC_AMINOACID_GROUP, 11, -0.04, RNRgb(0, 1, 1)       ),
  PDBAminoAcid( PDB_THR_AMINOACID, "Threonine",     "THR", 'T', PDB_HYDROXYLIC_AMINOACID_GROUP, 14,  0.26, RNRgb(0, 1, 1)       ),
  PDBAminoAcid( PDB_VAL_AMINOACID, "Valine",        "VAL", 'V', PDB_ALIPHATIC_AMINOACID_GROUP,  16,  1.22, RNRgb(1, 0.5, 0.4)   ),
  PDBAminoAcid( PDB_TRP_AMINOACID, "Tryptophan",    "TRP", 'W', PDB_AROMATIC_AMINOACID_GROUP,   24,  2.25, RNRgb(1, 0, 1)       ),
  PDBAminoAcid( PDB_TYR_AMINOACID, "Tyrosine",      "TYR", 'Y', PDB_AROMATIC_AMINOACID_GROUP,   21,  0.96, RNRgb(1, 0, 1)       )
};





PDBAminoAcid::
PDBAminoAcid(int id, const char *name, const char *code, char letter, int group, 
  int natoms, RNScalar hydrophobicity, const RNRgb& color)
  : id(id),
    letter(letter),
    group(group),
    natoms(natoms),
    hydrophobicity(hydrophobicity),
    color(color),
    mark(0),
    value(0),
    data(NULL)
{
  // Copy name
  if (name) { strncpy(this->name, name, 16); this->name[15] = 0; }
  else this->name[0] = '\0';

  // Copy code
  if (name) { strncpy(this->code, code, 4); this->code[3] = 0; }
  else this->code[0] = '\0';
}



PDBAminoAcid *
PDBFindAminoAcid(const char *name)
{
  // Search for amino acid with matching code
  for (int i = 0; i < PDBnaminoacids; i++) {
    PDBAminoAcid *aminoacid = &PDBaminoacids[i];
    if (!strcmp(name, aminoacid->code)) {
      return aminoacid;
    }
  }

  // Amino acid not found
  return NULL;
}





