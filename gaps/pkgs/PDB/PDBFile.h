// Header file for PDBFile class



// Class declaration

class PDBFile {
public:
  // Constructor
  PDBFile(const char *name = NULL);
  ~PDBFile(void);

  // Property functions
  int ID(void) const;
  const char *Name(void) const;
  const R3Box BBox(void) const;
  R3Point Centroid(void) const;
  RNLength Radius(void) const;
  const RNRgb& Color(void) const;
  RNBoolean IsMarked(void) const;
  RNScalar Value(void) const;
  void *Data(void) const;

  // Model access functions
  int NModels(void) const;
  PDBModel *Model(int k) const;

  // Search access functions
  PDBAtom *FindAtom(const char *str) const;
  PDBResidue *FindResidue(const char *str) const;
  PDBChain *FindChain(const char *str) const;
  PDBModel *FindModel(const char *str) const;
  PDBStructureType FindAny(const char *str, 
    PDBModel **model = NULL, PDBChain **chain = NULL, 
    PDBResidue **residue = NULL, PDBAtom **atom = NULL, 
    PDBStructureType maxlevel = PDB_ATOM) const;

  // Manipulation functions
  void Transform(const R3Affine& affine);
  PDBModel *InsertCopy(PDBModel *model);
  void SetName(const char *name);
  void SetData(void *data);
  void SetValue(RNScalar value);
  void SetMark(void);
  void UnsetMark(void);

  // I/O functions
  int ReadFile(const char *filename);
  int WriteFile(const char *filename) const;
  int Read(FILE *fp = NULL);
  int Write(FILE *fp = NULL) const;

  // More I/O functions
  int ReadASAFile(const char *filename);
  int ReadGrowFile(const char *filename);
  int ReadConsurfFiles(const char *hssp_basename);
  int ReadHsspFiles(const char *hssp_basename);
  int ReadJsdFiles(const char *jsd_basename,
    const char *conservation_file_source = "hssp.jsd",
    RNBoolean translate_renamed_chains = TRUE,
    RNBoolean translate_unnamed_chains = TRUE,
    RNBoolean translate_identical_chains = TRUE);
  int WriteJsdFiles(const char *jsd_basename,
    const char *conservation_file_source = "hssp.jsd");

  // ???
  void CreateBonds(PDBResidue *ligand, RNLength max_gap = 1);
  PDBFile *CopyBiomolecule(void);
  int IsBiomolecule(void) const;

public:
  // Data fields
  int id;
  char name[128];
  RNArray<char *> headers;
  RNArray<char *> trailers;
  RNArray<PDBModel *> models;
  RNMark mark;
  RNScalar value;
  void *data;
};



// Inline functions

inline int PDBFile::
ID(void) const
{
  // Return id
  return id;
}



inline const char *PDBFile::
Name(void) const
{
  // Return name
  return name;
}



inline RNBoolean PDBFile::
IsMarked(void) const
{
  // Return whether file is marked
  return (mark == PDBmark);
}



inline RNScalar PDBFile::
Value(void) const
{
  // Return scalar value associated with file by user (this is not used by the PDB package)
  return value;
}



inline void *PDBFile::
Data(void) const
{
  // Return data pointer associated with file by user (this is not used by the PDB package)
  return data;
}



inline int PDBFile::
NModels(void) const
{
  // Return number of models in file
  return models.NEntries();
}



inline PDBModel *PDBFile::
Model(int k) const
{
  // Return kth model in file
  return models.Kth(k);
}



inline void PDBFile::
SetName(const char *name)
{
  // Set name
  strncpy(this->name, name, 32);
  this->name[31] = '\0';
}



inline void PDBFile::
SetData(void *data)
{
  // Set data pointer associated with file by user (this is not used by the PDB package)
  this->data = data;
}



inline void PDBFile::
SetValue(RNScalar value)
{
  // Set scalar value associated with file by user (this is not used by the PDB package)
  this->value = value;
}



inline void PDBFile::
SetMark(void) 
{
  // Mark this file
  mark = PDBmark;
}



inline void PDBFile::
UnsetMark(void) 
{
  // Unmark this file
  mark = 0;
}



inline PDBAtom *PDBFile:: 
FindAtom(const char *str) const
{
  // Find atom matching string
  PDBAtom *atom = NULL;
  if (FindAny(str, NULL, NULL, NULL, &atom, PDB_ATOM) == PDB_ATOM) return atom;
  else return NULL;
}



inline PDBResidue *PDBFile:: 
FindResidue(const char *str) const
{
  // Find residue matching string
  PDBResidue *residue = NULL;
  if (FindAny(str, NULL, NULL, &residue, NULL, PDB_RESIDUE) == PDB_RESIDUE) return residue;
  else return NULL;
}



inline PDBChain *PDBFile:: 
FindChain(const char *str) const
{
  // Find chain matching string
  PDBChain *chain = NULL;
  if (FindAny(str, NULL, &chain, NULL, NULL, PDB_CHAIN) == PDB_CHAIN) return chain;
  else return NULL;
}



inline PDBModel *PDBFile::
FindModel(const char *str) const
{
  // Find model matching string
  PDBModel *model = NULL;
  if (FindAny(str, &model, NULL, NULL, NULL, PDB_MODEL) == PDB_MODEL) return model;
  else return NULL;
}



