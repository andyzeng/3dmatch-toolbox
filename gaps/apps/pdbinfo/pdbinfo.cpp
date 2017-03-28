// Source file for the pdb viewer program



// Include files 

#include "PDB/PDB.h"



// Program variables

static char *input_name = NULL;
static char *output_name = NULL;
static char *consurf_name = NULL;
static char *jsd_name = NULL;
static RNBoolean biomolecule = 0;
static RNBoolean print_models = 0;
static RNBoolean print_chains = 0;
static RNBoolean print_residues = 0;
static RNBoolean print_ligands = 0;
static RNBoolean print_atoms = 0;
static RNBoolean print_verbose = 0;



static PDBFile *
ReadPDB(char *filename)
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
    printf("  # Models = %d\n", file->NModels());
    fflush(stdout);
  }

  // Return success
  return file;
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
    printf("  # Residues = %d\n", nresidues);
    fflush(stdout);
  }

  // Return success
  return 1;
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
PrintInfo(PDBFile *file, const char *filename)
{
  // Open file
  FILE *fp = stdout;
  if (filename) {
    fp = fopen(filename, "w");
    if (!fp) {
      fprintf(stderr, "Unable to open info file %s\n", filename);
      return 0;
    }
  }

  if (print_verbose || print_models || print_chains || print_residues || print_ligands || print_atoms) {
    // Visit each model
    for (int i = 0; i < file->NModels(); i++) {
      PDBModel *model = file->Model(i);

      // Print model information
      if (print_verbose || print_models) {
        const R3Box& bbox = model->BBox();
        R3Point centroid = model->Centroid();
        RNLength max_radius = PDBMaxDistance(model->atoms, centroid);
        RNLength avg_radius = PDBAverageDistance(model->atoms, centroid);
        char is_biomolecule_string[32] = { 0 };
        int is_biomolecule = file->IsBiomolecule();
        if (is_biomolecule == 0) sprintf(is_biomolecule_string, "NO");
        else if (is_biomolecule == 1) sprintf(is_biomolecule_string, "YES");
        else sprintf(is_biomolecule_string, "UNKNOWN");
        fprintf(fp, "Model %s\n", model->Name());
        fprintf(fp, "  # Chains = %d\n", model->NChains());
        fprintf(fp, "  # Residues = %d\n", model->NResidues());
        fprintf(fp, "  # Atoms = %d\n", model->NAtoms());
        fprintf(fp, "  IsBiomolecule? = %s\n", is_biomolecule_string);
        fprintf(fp, "  Centroid = %g %g %g\n", centroid.X(), centroid.Y(), centroid.Z());
        fprintf(fp, "  Maximum Radius = %g\n", max_radius);
        fprintf(fp, "  Average Radius = %g\n", avg_radius);
        fprintf(fp, "  Bounding box = ( %g %g %g ) ( %g %g %g ) \n",
                bbox[0][0], bbox[0][1], bbox[0][2], 
                bbox[1][0], bbox[1][1], bbox[2][2]);
      }

      // Consider model contents
      if (print_verbose || print_chains || print_residues || print_ligands || print_atoms) {
        // Visit each chain
        for (int j = 0; j < model->NChains(); j++) {
          PDBChain *chain = model->Chain(j);

          // Print chain information
          if (print_verbose || print_chains) {
            const R3Box& bbox = chain->BBox();
            R3Point centroid = chain->Centroid();
            RNLength max_radius = PDBMaxDistance(chain->atoms, centroid);
            RNLength avg_radius = PDBAverageDistance(chain->atoms, centroid);
            fprintf(fp, "  Chain %s\n", (!strcmp(chain->Name(), " ")) ? "_" : chain->Name());
            fprintf(fp, "    # Residues = %d\n", chain->NResidues());
            fprintf(fp, "    # Atoms = %d\n", chain->NAtoms());
            fprintf(fp, "    Centroid = %g %g %g\n", centroid.X(), centroid.Y(), centroid.Z());
            fprintf(fp, "    Maximum Radius = %g\n", max_radius);
            fprintf(fp, "    Average Radius = %g\n", avg_radius);
            fprintf(fp, "    Bounding box = ( %g %g %g ) ( %g %g %g ) \n",
                    bbox[0][0], bbox[0][1], bbox[0][2], 
                    bbox[1][0], bbox[1][1], bbox[2][2]);
          }

          // Consider chain contents
          if (print_verbose || print_residues || print_ligands || print_atoms) {
            // Visit each residue
            for (int k = 0; k < chain->NResidues(); k++) {
              PDBResidue *residue = chain->Residue(k);

              // Print residue information
              if (print_verbose || (print_residues && !residue->HasHetAtoms()) || (print_ligands && residue->HasHetAtoms())) {
                PDBAminoAcid *aminoacid = residue->AminoAcid();
                int icode = residue->InsertionCode();
                const R3Box& bbox = residue->BBox();
                R3Point centroid = residue->Centroid();
                RNLength max_radius = PDBMaxDistance(residue->atoms, centroid);
                RNLength avg_radius = PDBAverageDistance(residue->atoms, centroid);
                RNScalar conservation = residue->conservation;
                if (conservation == PDB_UNKNOWN) conservation = -1;
                fprintf(fp, "    Residue %s ...\n", residue->Name());
                fprintf(fp, "      # Atoms = %d\n", residue->NAtoms());
                fprintf(fp, "      Amino Acid = %s\n", (aminoacid) ? aminoacid->Name() : "Unknown");
                fprintf(fp, "      Sequence = %d\n", residue->Sequence());
                fprintf(fp, "      InsertionCode = %c\n", (icode == ' ') ? '_' : icode);
                fprintf(fp, "      Centroid = %g %g %g\n", centroid.X(), centroid.Y(), centroid.Z());
                fprintf(fp, "      Conservation = %g\n", conservation);
                fprintf(fp, "      Maximum Radius = %g\n", max_radius);
                fprintf(fp, "      Average Radius = %g\n", avg_radius);
                fprintf(fp, "      Bounding box = ( %g %g %g ) ( %g %g %g ) \n",
                        bbox[0][0], bbox[0][1], bbox[0][2], 
                        bbox[1][0], bbox[1][1], bbox[2][2]);
                fprintf(fp, "      String = %s-%s-%s-%s-%d-%c\n", 
                        file->Name(), model->Name(), (strcmp(chain->Name(), " ")) ? chain->Name() : "_",
                        residue->Name(), residue->Sequence(), (icode == ' ') ? '_' : icode); 
              }

              // Consider residue contents
              if (print_verbose || print_atoms) {
                // Visit each atom
                for (int m = 0; m < residue->NAtoms(); m++) {
                  PDBAtom *atom = residue->Atom(m);

                  // Print atom information
                  if (print_verbose || print_atoms) {
                    const R3Point& p = atom->Position();
                    PDBElement *element = atom->Element();
                    int altloc = atom->AlternateLocation();
                    fprintf(fp, "      Atom %s ...\n", atom->Name());
                    fprintf(fp, "        Hetatm = %d\n", (atom->IsHetAtom()) ? 1 : 0);
                    fprintf(fp, "        Serial = %d\n", atom->Serial());
                    fprintf(fp, "        Element = %s\n", (element) ? element->Name() : "Unknown");
                    fprintf(fp, "        Position = %g %g %g\n", p.X(), p.Y(), p.Z());
                    fprintf(fp, "        Alternate Location = %c\n", (altloc == ' ') ? '_' : altloc);
                    fprintf(fp, "        Radius = %g\n", atom->Radius());
                    fprintf(fp, "        Occupancy = %g\n", atom->Occupancy());
                    fprintf(fp, "        Temperature Factor = %g\n", atom->TempFactor());
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Close file
  if (filename) fclose(fp);

  // Return success
  return 1;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if (argc < 2) {
    fprintf(stderr, "Usage: pdbinfo pdbfile [options]\n");
    return 0;
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1; 
      else if (!strcmp(*argv, "-biomolecule")) biomolecule = 1; 
      else if (!strcmp(*argv, "-models")) print_models = 1; 
      else if (!strcmp(*argv, "-chains")) print_chains = 1; 
      else if (!strcmp(*argv, "-residues")) print_residues = 1; 
      else if (!strcmp(*argv, "-ligands")) print_ligands = 1; 
      else if (!strcmp(*argv, "-atoms")) print_atoms = 1; 
      else if (!strcmp(*argv, "-consurf")) { argc--; argv++; consurf_name = *argv; }
      else if (!strcmp(*argv, "-jsd")) { argc--; argv++; jsd_name = *argv; }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    }
    else {
      if (!input_name) input_name = *argv;
      else if (!output_name) output_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    }
    argv++; argc--;
  }

  // Check input filename
  if (!input_name) {
    fprintf(stderr, "You did not specify an input pdb file.\n");
    return 0;
  }

  // Check arguments
  if (!print_verbose && !print_models && !print_chains && !print_residues && !print_ligands && !print_atoms) {
    print_models = TRUE;
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
  PDBFile *file = ReadPDB(input_name);
  if (!file) exit(-1);

  // Read conservation files
  if (jsd_name) {
    int status = ReadJsdFiles(file, jsd_name);
    if (!status) exit(-1);
  }
  else if (consurf_name) {
    int status = ReadConsurfFiles(file, consurf_name);
    if (!status) exit(-1);
  }

  // Print info about PDB file
  int status = PrintInfo(file, output_name);
  if (!status) exit(-1);

  // Return success
  return 0;
}
