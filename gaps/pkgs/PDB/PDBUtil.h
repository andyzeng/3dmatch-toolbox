// Include file for PDB utility functions



// Low-level functions for computing alignment for arrays of atoms

R3Point PDBCentroid(const RNArray<PDBAtom *>& atoms);
R3Box PDBBox(const RNArray<PDBAtom *>& atoms);
RNLength PDBMaxDistance(const RNArray<PDBAtom *>& atoms, const R3Point& center);
RNLength PDBAverageDistance(const RNArray<PDBAtom *>& atoms, const R3Point& center);
R3Triad PDBPrincipleAxes(const RNArray<PDBAtom *>& atoms, const R3Point& centroid);
R3Affine PDBAlignmentTransformation(const RNArray<PDBAtom *>& atoms,
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
R3Affine PDBAlignmentTransformation(const RNArray<PDBAtom *>& atoms1, const RNArray<PDBAtom *>& atoms2, RNScalar* weights = NULL, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
RNScalar PDBMeanAlignmentError(const RNArray<PDBAtom *>& atoms1, const RNArray<PDBAtom *>& atoms2, RNScalar* weights, const R3Affine& affine);




// Functions for computing alignment for models, chains, residues with canonical coordinate system

R3Affine PDBAlignmentTransformation(const PDBModel *model, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
R3Affine PDBAlignmentTransformation(const PDBChain *chain, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
R3Affine PDBAlignmentTransformation(const PDBResidue *residue, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);



// Functions for aligning models, chains, residues with canonical coordinate system (also applies aligning transformation)

void PDBAlignModel(PDBModel *model, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
void PDBAlignChain(PDBChain *chain, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
void PDBAlignResidue(PDBResidue *residue, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);



// Functions for computing alignment for pairs of models, chains, residues

R3Affine PDBAlignmentTransformation(const PDBModel *model1, const PDBModel *model2, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
R3Affine PDBAlignmentTransformation(const PDBChain *chain1, const PDBChain *chain2, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
R3Affine PDBAlignmentTransformation(const PDBResidue *residue1, const PDBChain *residue2, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);



// Functions for aligning models, chains, residues (also applies aligning transformation)

void PDBAlignModel(PDBModel *model1, const PDBModel *model2, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
void PDBAlignChain(PDBChain *chain1, const PDBChain *chain2, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);
void PDBAlignResidue(PDBResidue *residue1, const PDBResidue *residue2, 
  RNBoolean align_translation = TRUE, RNBoolean align_rotation = TRUE, RNBoolean align_scale = FALSE);




