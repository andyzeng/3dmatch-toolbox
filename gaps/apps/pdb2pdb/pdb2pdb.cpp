// Source file for the pdb viewer program



// Include files 

#include "PDB/PDB.h"



// Program variables

static char *input_name = NULL;
static char *output_name = NULL;
static char *jsd_name = NULL;
static char *consurf_name = NULL;
static char *asa_name = NULL;
static char *grow_name = NULL;
static RNArray<char *> atom_names;
static RNArray<char *> residue_names;
static RNArray<char *> chain_names;
static RNArray<char *> model_names;
static RNBoolean biomolecule = 0;
static RNBoolean select_all = 1;
static RNBoolean select_hetatoms = 0;
static RNScalar min_conservation = 0;
static const int max_contacts = 64;
static char *contact_names[max_contacts];
static PDBStructureType contact_granularities[max_contacts];
static int ncontacts = 0;
static const int max_spheres = 64;
static char *sphere_names[max_spheres];
static R3Point sphere_centers[max_spheres];
static RNScalar sphere_radius[max_spheres];
static PDBStructureType sphere_granularities[max_spheres];
static int nspheres = 0;
static const int max_offsets = 64;
static char *offset_names[max_offsets];
static RNScalar offset_radius[max_offsets];
static PDBStructureType offset_granularities[max_offsets];
static int noffsets = 0;
static R3Affine *transformation = NULL;
static RNBoolean remove_remarks = FALSE;
static int print_verbose = 0;



static PDBFile *
ReadPDBFile(char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate PDBFile
  PDBFile *file = new PDBFile(filename);
  if (!file) {
    RNFail("Unable to allocate PDB file for %s", filename);
    return NULL;
  }

  // Read PDB file
  if (!file->ReadFile(filename)) {
    RNFail("Unable to read PDB file %s", filename);
    return NULL;
  }

  // Get biomolecule
  if (biomolecule && (!file->IsBiomolecule())) {
    PDBFile *file2 = file->CopyBiomolecule();
    delete file;
    file = file2;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read PDB file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    for (int i = 0; i < file->NModels(); i++) {
      printf("  Model %d ...\n", i);
      printf("  # Chains = %d\n", file->Model(i)->NChains());
      printf("  # Residues = %d\n", file->Model(i)->NResidues());
      printf("  # Atoms = %d\n", file->Model(i)->NAtoms());
    }
    fflush(stdout);
  }

  // Return success
  return file;
}



static int
ReadJsdFiles(PDBFile *file, char *jsd_basename)
{
  // Check number of models
  if (file->NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", jsd_basename);
    return 0;
  }

  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read conservation files
  int nresidues = file->ReadJsdFiles(jsd_basename);

  // Print statistics
  if (print_verbose && (nresidues > 0)) {
    printf("Read jsd files ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf(" # Residues = %d\n", nresidues);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadConsurfFiles(PDBFile *file, char *consurf_basename)
{
  // Check number of models
  if (file->NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", consurf_basename);
    return 0;
  }

  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read conservation files
  int nresidues = file->ReadConsurfFiles(consurf_basename);

  // Print statistics
  if (print_verbose && (nresidues > 0)) {
    printf("Read consurf files ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf(" # Residues = %d\n", nresidues);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadASAFile(PDBFile *file, char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read ASA file
  int natoms = file->ReadASAFile(filename);
  if (natoms == 0) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Read accessible surface area file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Surface Atoms = %d\n", natoms);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadGrowFile(PDBFile *file, char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read grow file
  int nbonds = file->ReadGrowFile(filename);
  if (nbonds == 0) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Read grow file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Bonds = %d\n", nbonds);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
SelectAtoms(PDBFile *file, RNArray<PDBAtom *>& atoms)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get model for file
  if (file->NModels() == 0) return 0;
  PDBModel *model = file->Model(0);

  // Check if should select all atoms
  if (select_all) {
    atoms.Append(model->atoms);
    return atoms.NEntries();
  }

  // Select hetatoms
  if (select_hetatoms) {
    // Select all hetatoms
    for (int j = 0; j < model->NAtoms(); j++) {
      PDBAtom *atom = model->Atom(j);
      if (!atom->IsHetAtom()) continue;
      atom->SetMark();
    }
  }

  // Select atoms
  for (int i = 0; i < atom_names.NEntries(); i++) {
    // Find atom in file
    PDBAtom *atom = file->FindAtom(atom_names[i]);
    if (!atom) {
      fprintf(stderr, "Unable to find atom \"%s\" in %s\n", atom_names[i], file->Name());
      return 0;
    }

    // Mark atom
    atom->SetMark();
  }

  // Select residues
  for (int i = 0; i < residue_names.NEntries(); i++) {
    // Find residue in file
    PDBResidue *residue = file->FindResidue(residue_names[i]);
    if (!residue) {
      fprintf(stderr, "Unable to find residue \"%s\" in %s\n", residue_names[i], file->Name());
      return 0;
    }

    // Mark residue
    residue->SetMark();
  }

  // Select chains
  for (int i = 0; i < chain_names.NEntries(); i++) {
    // Find chain in file
    PDBChain *chain = file->FindChain(chain_names[i]);
    if (!chain) {
      fprintf(stderr, "Unable to find chain \"%s\" in %s\n", chain_names[i], file->Name());
      return 0;
    }

    // Mark chain
    chain->SetMark();
  }

  // Select models
  for (int i = 0; i < model_names.NEntries(); i++) {
    // Find model in file
    PDBModel *model = file->FindModel(model_names[i]);
    if (!model) {
      fprintf(stderr, "Unable to find model \"%s\" in %s\n", model_names[i], file->Name());
      return 0;
    }

    // Mark model
    model->SetMark();
  }

  // Mark bonding atoms
  if (ncontacts > 0) {
    // Make array of contacted atoms
    RNArray<PDBAtom *> contact_atoms;
    for (int i = 0; i < ncontacts; i++) {
      PDBAtom *atom;
      PDBResidue *residue;
      PDBChain *chain;
      PDBModel *model;
      PDBStructureType result = file->FindAny(contact_names[i], &model, &chain, &residue, &atom);
      if (result == PDB_MODEL) { for (int j = 0; j < model->NAtoms(); j++) contact_atoms.Insert(model->Atom(j)); }
      else if (result == PDB_CHAIN) { for (int j = 0; j < chain->NAtoms(); j++) contact_atoms.Insert(chain->Atom(j)); }
      else if (result == PDB_RESIDUE) { for (int j = 0; j < residue->NAtoms(); j++) contact_atoms.Insert(residue->Atom(j)); }
      else if (result == PDB_ATOM) { contact_atoms.Insert(atom); }
      else {
        fprintf(stderr, "Unable to find contact \"%s\" in %s\n", contact_names[i], file->Name());
        return 0;
      }
    }

    // Mark contacting atoms
    for (int i = 0; i < contact_atoms.NEntries(); i++) {
      PDBAtom *contact = contact_atoms.Kth(i);
      for (int j = 0; j < contact->NBonds(); j++) {
        PDBBond *bond = contact->Bond(j);
        PDBAtom *atom = bond->OtherAtom(contact);
        if (contact_granularities[i] == PDB_ATOM) atom->SetMark();
        else if (contact_granularities[i] == PDB_RESIDUE) atom->Residue()->SetMark();
        else if (contact_granularities[i] == PDB_CHAIN) atom->Chain()->SetMark();
        else if (contact_granularities[i] == PDB_MODEL) atom->Model()->SetMark();
        else RNAbort("Invalid contact granularity");
      }
    }
  }

  // Mark atoms touching a sphere centered on structure
  for (int i = 0; i < nspheres; i++) {
    // Determine radius
    RNLength radius = sphere_radius[i];

    // Determine center
    if (sphere_names[i]) {
      PDBAtom *atom;
      PDBResidue *residue;
      PDBChain *chain;
      PDBModel *model;
      PDBStructureType result = file->FindAny(sphere_names[i], &model, &chain, &residue, &atom);
      if (result == PDB_MODEL) { sphere_centers[i] = model->Centroid(); }
      else if (result == PDB_CHAIN) { sphere_centers[i] = chain->Centroid(); }
      else if (result == PDB_RESIDUE) { sphere_centers[i] = residue->Centroid(); }
      else if (result == PDB_ATOM) { sphere_centers[i] = atom->Position(); }
      else {
        fprintf(stderr, "Unable to find \"%s\" in %s\n", sphere_names[i], file->Name());
        return 0;
      }
    }

    // Mark atoms touching sphere
    for (int j = 0; j < model->NAtoms(); j++) {
      PDBAtom *atom = model->Atom(j);
      if (atom->IsHetAtom()) continue;
      if (atom->accessible_surface_area == 0) continue;
      if (R3Distance(sphere_centers[i], atom->Position()) > radius + atom->Radius()) continue;
      if (sphere_granularities[i] == PDB_ATOM) atom->SetMark();
      else if (sphere_granularities[i] == PDB_RESIDUE) atom->Residue()->SetMark();
      else if (sphere_granularities[i] == PDB_CHAIN) atom->Chain()->SetMark();
      else if (sphere_granularities[i] == PDB_MODEL) atom->Model()->SetMark();
      else RNAbort("Invalid sphere granularity");
    }
  }

  // Mark atoms within offset of structure
  for (int i = 0; i < noffsets; i++) {
    // Make array of atoms to form centers of offset
    RNArray<PDBAtom *> offset_atoms;
    for (int j = 0; j < noffsets; j++) {
      PDBAtom *atom;
      PDBResidue *residue;
      PDBChain *chain;
      PDBModel *model;
      PDBStructureType result = file->FindAny(offset_names[j], &model, &chain, &residue, &atom);
      if (result == PDB_MODEL) { for (int j = 0; j < model->NAtoms(); j++) offset_atoms.Insert(model->Atom(j)); }
      else if (result == PDB_CHAIN) { for (int j = 0; j < chain->NAtoms(); j++) offset_atoms.Insert(chain->Atom(j)); }
      else if (result == PDB_RESIDUE) { for (int j = 0; j < residue->NAtoms(); j++) offset_atoms.Insert(residue->Atom(j)); }
      else if (result == PDB_ATOM) { offset_atoms.Insert(atom); }
      else {
        fprintf(stderr, "Unable to find center \"%s\" in %s\n", offset_names[i], file->Name());
        return 0;
      }
    }

    // Mark atoms within offset of any center atom
    for (int j = 0; j < model->NAtoms(); j++) {
      PDBAtom *atom = model->Atom(j);
      if (atom->IsHetAtom()) continue;
      if (atom->accessible_surface_area == 0) continue;
      for (int k = 0; k < offset_atoms.NEntries(); k++) {
        PDBAtom *center = offset_atoms.Kth(k);
        RNLength distance = R3Distance(atom->Position(), center->Position());
        if (distance > offset_radius[i] + atom->Radius() + center->Radius()) continue;
        if (offset_granularities[i] == PDB_ATOM) atom->SetMark();
        else if (offset_granularities[i] == PDB_RESIDUE) atom->Residue()->SetMark();
        else if (offset_granularities[i] == PDB_CHAIN) atom->Chain()->SetMark();
        else if (offset_granularities[i] == PDB_MODEL) atom->Model()->SetMark();
        else RNAbort("Invalid offset granularity");
      }
    }
  }

  // Flatten selection into array of atoms
  RNBoolean include = FALSE;
  for (int i = 0; i < file->NModels(); i++) {
    PDBModel *model = file->Model(i);

    // Check mark
    RNBoolean file_include = include;
    if (model->IsMarked()) include = TRUE;

    // Select chains
    for (int j = 0; j < model->NChains(); j++) {
      PDBChain *chain = model->Chain(j);

      // Check mark
      RNBoolean model_include = include;
      if (chain->IsMarked()) include = TRUE;

      // Insert residues from chain
      for (int k = 0; k < chain->NResidues(); k++) {
        PDBResidue *residue = chain->Residue(k);

        // Check mark
        RNBoolean chain_include = include;
        if (residue->IsMarked()) include = TRUE;

        // Check conservation
        if ((min_conservation > 0) && (residue->conservation != PDB_UNKNOWN)) {
          if (residue->conservation < min_conservation) continue;
        }

        // Insert atoms from residue
        for (int m = 0; m < residue->NAtoms(); m++) {
          PDBAtom *atom = residue->Atom(m);
          if (atom->IsMarked() || include) {
            atoms.Insert(atom);
          }
        }

        // Restore include
        include = chain_include;
      }

      // Restore include
      include = model_include;
    }

    // Restore include
    include = file_include;
  }

  // Check atoms
  if (atoms.IsEmpty()) {
    fprintf(stderr, "No atoms selected.\n");
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("Selected atoms ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Atoms = %d\n", atoms.NEntries());
    fflush(stdout);
  }

  // Return number of atoms
  return atoms.NEntries();
}



static PDBFile *
CreatePDB(PDBFile *file1, const RNArray<PDBAtom *>& atoms)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate new PDB file
  PDBFile *file2 = new PDBFile("");
  assert(file2);

  // Copy file stuff
  if (!remove_remarks) {
    for (int i = 0; i < file1->headers.NEntries(); i++)  
      file2->headers.Insert(strdup(file1->headers[i]));
    for (int i = 0; i < file1->trailers.NEntries(); i++) 
      file2->trailers.Insert(strdup(file1->trailers[i]));
  }

  // Copy atoms from file1 into file2
  int atom_count = 0;
  for (int i = 0; i < atoms.NEntries(); i++) {
    PDBAtom *atom1 = atoms.Kth(i);

    // Find residue in file1
    PDBResidue *residue1 = atom1->Residue();
    if (!residue1) { 
      fprintf(stderr, "Atom %s has no residue?\n", atom1->Name()); 
      return 0; 
    }

    // Find chain in file1
    PDBChain *chain1 = residue1->Chain();
    if (!chain1) { 
      fprintf(stderr, "Residue %s has no chain?\n", residue1->Name()); 
      return 0;
    }

    // Find model in file1
    PDBModel *model1 = chain1->Model();
    if (!model1) { 
      fprintf(stderr, "Chain %s has no model?\n", chain1->Name()); 
      return 0;
    }

    // Get/create model in file2
    PDBModel *model2 = (PDBModel *) model1->Data();
    if (!model2) {
      model2 = new PDBModel(file2, model1->Name());
      model1->SetData(model2);
      assert(model2);
    }

    // Get/create chain in model2
    PDBChain *chain2 = (PDBChain *) chain1->Data();
    if (!chain2) {
      chain2 = new PDBChain(model2, chain1->Name());
      chain1->SetData(chain2);
      assert(chain2);
    }

    // Get/create residue in model2
    PDBResidue *residue2 = (PDBResidue *) residue1->Data();
    if (!residue2) {
      residue2 = new PDBResidue(model2, chain2, residue1->Name(), residue1->Sequence(), residue1->InsertionCode());
      residue1->SetData(residue2);
      assert(residue2);
    }

    // Get/create atom in model2
    PDBAtom *atom2 = (PDBAtom *) atom1->Data();
    if (!atom2) {
      atom2 = new PDBAtom(model2, chain2, residue2, atom1->Element(), 
                          atom1->Serial(), atom1->Name(), atom1->AlternateLocation(),
                          atom1->Position().X(), atom1->Position().Y(), atom1->Position().Z(), 
                          atom1->Occupancy(), atom1->TempFactor(), atom1->Charge(), atom1->IsHetAtom());
      atom1->SetData(atom2);
      assert(atom2);
      atom_count++;
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("Created new PDB file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Atoms = %d\n", atom_count);
    fflush(stdout);
  }

  // Return new file
  return file2;
}



static int 
WritePDB(PDBFile *file, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write PDB file
  int status = file->WriteFile(filename);
  if (!status) {
    fprintf(stderr, "Unable to write %s\n", filename);
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("Wrote PDB file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    for (int i = 0; i < file->NModels(); i++) {
      printf("  Model %d ...\n", i);
      printf("  # Chains = %d\n", file->Model(i)->NChains());
      printf("  # Residues = %d\n", file->Model(i)->NResidues());
      printf("  # Atoms = %d\n", file->Model(i)->NAtoms());
    }
    fflush(stdout);
  }

  // Return status
  return status;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if (argc < 3) {
    printf("Usage: pdb2pdb inputfile outputfile [growfile] [-atom name] [-residue name] [-chain name] [-model name] [-bonding name] [-v]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1; 
      else if (!strcmp(*argv, "-remove_remarks")) remove_remarks = 1; 
      else if (!strcmp(*argv, "-biomolecule")) biomolecule = 1; 
      else if (!strcmp(*argv, "-jsd")) { argc--; argv++; jsd_name = *argv; }
      else if (!strcmp(*argv, "-consurf")) { argc--; argv++; consurf_name = *argv; }
      else if (!strcmp(*argv, "-asa")) { argc--; argv++; asa_name = *argv; }
      else if (!strcmp(*argv, "-grow")) { argc--; argv++; grow_name = *argv; }
      else if (!strcmp(*argv, "-hetatoms")) { select_hetatoms = 1; select_all = FALSE; }
      else if (!strcmp(*argv, "-atom")) { argc--; argv++; atom_names.Insert(*argv); select_all = FALSE; }
      else if (!strcmp(*argv, "-residue")) { argc--; argv++; residue_names.Insert(*argv); select_all = FALSE; }
      else if (!strcmp(*argv, "-chain")) { argc--; argv++; chain_names.Insert(*argv); select_all = FALSE; }
      else if (!strcmp(*argv, "-model")) { argc--; argv++; model_names.Insert(*argv); select_all = FALSE; }
      else if (!strcmp(*argv, "-bonding")) { 
        if (ncontacts < max_contacts) {
          select_all = FALSE; 
          argc--; argv++; contact_names[ncontacts] = *argv;
          argc--; argv++; 
          if (!strcmp(*argv, "atom")) contact_granularities[ncontacts] = PDB_ATOM;
          else if (!strcmp(*argv, "residue")) contact_granularities[ncontacts] = PDB_RESIDUE;
          else if (!strcmp(*argv, "chain")) contact_granularities[ncontacts] = PDB_CHAIN;
          else if (!strcmp(*argv, "model")) contact_granularities[ncontacts] = PDB_MODEL;
          else { fprintf(stderr, "Invalid contact granularity: %s\n", *argv); exit(-1); }
          ncontacts++;
        }
      }
      else if (!strcmp(*argv, "-sphere")) { 
        if (nspheres < max_spheres) {
          select_all = FALSE; 
          argc--; argv++; char *bufferp = strchr(*argv, '-');
          if ((bufferp) && (bufferp != *argv)) {
            // "-sphere <name> ..."
            sphere_names[nspheres] = *argv;
            sphere_centers[nspheres] = R3zero_point;
          }
          else { 
            // "-sphere <x> <y> <z> ..."
            RNScalar x = atof(*argv);
            argc--; argv++; RNScalar y = atof(*argv);
            argc--; argv++; RNScalar z = atof(*argv);
            sphere_names[nspheres] = NULL;
            sphere_centers[nspheres] = R3Point(x, y, z);
          }
          argc--; argv++; sphere_radius[nspheres] = atof(*argv); 
          argc--; argv++; 
          if (!strcmp(*argv, "atom")) sphere_granularities[nspheres] = PDB_ATOM;
          else if (!strcmp(*argv, "residue")) sphere_granularities[nspheres] = PDB_RESIDUE;
          else if (!strcmp(*argv, "chain")) sphere_granularities[nspheres] = PDB_CHAIN;
          else if (!strcmp(*argv, "model")) sphere_granularities[nspheres] = PDB_MODEL;
          else { fprintf(stderr, "Invalid sphere granularity: %s\n", *argv); exit(-1); }
          nspheres++;
        }
      }
      else if (!strcmp(*argv, "-offset")) { 
        if (noffsets < max_offsets) {
          select_all = FALSE; 
          argc--; argv++; offset_names[noffsets] = *argv;
          argc--; argv++; offset_radius[noffsets] = atof(*argv); 
          argc--; argv++; 
          if (!strcmp(*argv, "atom")) offset_granularities[noffsets] = PDB_ATOM;
          else if (!strcmp(*argv, "residue")) offset_granularities[noffsets] = PDB_RESIDUE;
          else if (!strcmp(*argv, "chain")) offset_granularities[noffsets] = PDB_CHAIN;
          else if (!strcmp(*argv, "model")) offset_granularities[noffsets] = PDB_MODEL;
          else { fprintf(stderr, "Invalid offset granularity: %s\n", *argv); exit(-1); }
          noffsets++;
        }
      }
      else if (!strcmp(*argv, "-matrix")) { 
        R4Matrix m;
        argc--; argv++; m[0][0] = atof(*argv);
        argc--; argv++; m[0][1] = atof(*argv);
        argc--; argv++; m[0][2] = atof(*argv);
        argc--; argv++; m[0][3] = atof(*argv);
        argc--; argv++; m[1][0] = atof(*argv);
        argc--; argv++; m[1][1] = atof(*argv);
        argc--; argv++; m[1][2] = atof(*argv);
        argc--; argv++; m[1][3] = atof(*argv);
        argc--; argv++; m[2][0] = atof(*argv);
        argc--; argv++; m[2][1] = atof(*argv);
        argc--; argv++; m[2][2] = atof(*argv);
        argc--; argv++; m[2][3] = atof(*argv);
        argc--; argv++; m[3][0] = atof(*argv);
        argc--; argv++; m[3][1] = atof(*argv);
        argc--; argv++; m[3][2] = atof(*argv);
        argc--; argv++; m[3][3] = atof(*argv);
        transformation = new R3Affine(m);
      }
      else { 
        fprintf(stderr, "Invalid program argument: %s\n", *argv); 
        return 0;
      }
    }
    else {
      if (!input_name) input_name = *argv;
      else if (!output_name) output_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s\n", *argv); return 0; }
    }
    argv++; argc--;
  }

  // Check input filename
  if (!input_name) {
    fprintf(stderr, "You did not specify an input pdb file.\n");
    return 0;
  }

  // Check output filename
  if (!output_name) {
    fprintf(stderr, "You did not specify an output pdb file.\n");
    return 0;
  }

  // Check grow filename
  if (!grow_name && ncontacts) {
    fprintf(stderr, "You must provide a grow file if you specify bonding structures\n");
    return 0;
  }

  // Return OK status 
  return 1;
}


    
int 
main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read PDB file
  PDBFile *file1 = ReadPDBFile(input_name);
  if (!file1) exit(-1);

  // Read conservation files
  if (jsd_name) {
    int status = ReadJsdFiles(file1, jsd_name);
    if (!status) exit(-1);
  }
  else if (consurf_name) {
    int status = ReadConsurfFiles(file1, consurf_name);
    if (!status) exit(-1);
  }

  // Read accessible surface area file
  if (asa_name) {
    int status = ReadASAFile(file1, asa_name);
    if (!status) exit(-1);
  }

  // Read grow file
  if (grow_name) {
    int status = ReadGrowFile(file1, grow_name);
    if (!status) exit(-1);
  }

  // Select atoms from PDB file
  RNArray<PDBAtom *> atoms;
  int status1 = SelectAtoms(file1, atoms);
  if (!status1) exit(-1);

  // Create PDB file containing subset of residues
  PDBFile *file2 = CreatePDB(file1, atoms);
  if (!file2) exit(-1);

  // Apply transformation to PDB file
  if (transformation) file2->Transform(*transformation);

  // Write PDB file
  int status2 = WritePDB(file2, output_name);
  if (!status2) exit(-1);

  // Return success
  return 0;
}



