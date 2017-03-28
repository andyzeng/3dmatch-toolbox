// Source file for the pdb viewer program



// Include files 

#include "R3Shapes/R3Shapes.h"
#include "PDB/PDB.h"



// Program variables

static char *pdb_name = NULL;
static char *grid_name = NULL;
static char *ligand_name = NULL;
static char *grow_name = NULL;
static char *asa_name = NULL;
static char *consurf_name = NULL;
static int site_type = 0; // 0=protein, 1=ligand
static int rasterization_type = 0;  // 0=points, 1=solid, 2=surface, 3=dots, 4=offset
static int normalization_type = 0;  
static int grid_resolution[3] = { 0, 0, 0 };
static int max_grid_resolution = 256;
static RNLength grid_spacing = 0;
static R3Point *world_center = NULL;
static RNScalar world_radius = 0;
static RNScalar world_border = 0;
static RNScalar sigma = 0;
static RNScalar threshold = PDB_UNKNOWN;
static RNScalar offset = 0;
static RNBoolean elements = FALSE;
static RNBoolean conservation = FALSE;
static RNBoolean hydrophobicity = FALSE;
static RNBoolean charge = FALSE;
static RNBoolean accessible_surface_area = FALSE;
static RNBoolean ignore_zeros_when_blur = FALSE;
static int print_verbose = 0;



// Local constants

const RNScalar DONT_CARE = -56789;



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
  if (!file->ReadFile(filename)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Read PDB file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    for (int i =0; i < file->NModels(); i++) {
      PDBModel *model = file->Model(i);
      printf("  Model %s ...\n", model->Name());
      printf("  # Chains = %d\n", model->NChains());
      printf("  # Residues = %d\n", model->NResidues());
      printf("  # Atoms = %d\n", model->NAtoms());
    }
    fflush(stdout);
  }

  // Return success
  return file;
}



static PDBResidue *
FindLigand(PDBFile *file, const char *ligand_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Find ligand
  PDBResidue *ligand = file->FindResidue(ligand_name);
  if (!ligand) {
    fprintf(stderr, "Unable to find ligand %s in %s\n", ligand_name, pdb_name); 
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("Found ligand ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Atoms = %d\n", ligand->NAtoms());
    fflush(stdout);
  }

  // Return ligand
  return ligand;
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



void
BlurGrid(R3Grid *grid, RNLength world_sigma) 
{
  // Build filter
  RNScalar sigma = world_sigma * grid->WorldToGridScaleFactor();
  int filter_radius = (int) (3 * sigma);
  RNScalar *filter = new RNScalar [ filter_radius + 1 ];
  assert(filter);

  // Make buffer for temporary copy of row
  int res = grid->XResolution();
  if (res < grid->YResolution()) res = grid->YResolution();
  if (res < grid->ZResolution()) res = grid->ZResolution();
  RNScalar *buffer = new RNScalar [ res ];
  assert(buffer);

  // Fill filter with Gaussian 
  const RNScalar sqrt_two_pi = sqrt(RN_TWO_PI);
  double a = sqrt_two_pi * sigma;
  double fac = 1.0 / (a * a * a);
  double denom = 2.0 * sigma * sigma;
  for (int i = 0; i <= filter_radius; i++) {
    filter[i] = fac * exp(-i * i / denom);
  }

  // Convolve grid with filter in X direction
  for (int k = 0; k < grid->ZResolution(); k++) {
    for (int j = 0; j < grid->YResolution(); j++) { 
      for (int i = 0; i < grid->XResolution(); i++) 
        buffer[i] = grid->GridValue(i, j, k); 
      for (int i = 0; i < grid->XResolution(); i++) { 
        RNScalar sum = 0.0;
        RNScalar weight = 0;
        RNScalar value = buffer[i];
        if (value != DONT_CARE) {
          sum += filter[0] * value;
          weight += filter[0];
        }
        int nsamples = i;
        if (nsamples > filter_radius) nsamples = filter_radius;
        for (int m = 1; m <= nsamples; m++) {
          RNScalar value = buffer[i - m];
          if (value != DONT_CARE) {
            sum += filter[m] * value;
            weight += filter[m];
          }
        }
        nsamples = grid->XResolution() - 1 - i;
        if (nsamples > filter_radius) nsamples = filter_radius;
        for (int m = 1; m <= nsamples; m++) {
          RNScalar value = buffer[i + m];
          if (value != DONT_CARE) {
            sum += filter[m] * value;
            weight += filter[m];
          }
        }
        if (weight > 0) grid->SetGridValue(i, j, k, sum / weight);
      }
    }
  }

  // Convolve grid with filter in Y direction
  for (int k = 0; k < grid->ZResolution(); k++) {
    for (int j = 0; j < grid->XResolution(); j++) { 
      for (int i = 0; i < grid->YResolution(); i++) 
        buffer[i] = grid->GridValue(j, i, k); 
      for (int i = 0; i < grid->YResolution(); i++) { 
        RNScalar sum = 0.0;
        RNScalar weight = 0;
        RNScalar value = buffer[i];
        if (value != DONT_CARE) {
          sum += filter[0] * value;
          weight += filter[0];
        }
        int nsamples = i;
        if (nsamples > filter_radius) nsamples = filter_radius;
        for (int m = 1; m <= nsamples; m++) {
          RNScalar value = buffer[i - m];
          if (value != DONT_CARE) {
            sum += filter[m] * value;
            weight += filter[m];
          }
        }
        nsamples = grid->YResolution() - 1 - i;
        if (nsamples > filter_radius) nsamples = filter_radius;
        for (int m = 1; m <= nsamples; m++) {
          RNScalar value = buffer[i + m];
          if (value != DONT_CARE) {
            sum += filter[m] * value;
            weight += filter[m];
          }
        }
        if (weight > 0) grid->SetGridValue(j, i, k, sum / weight);
      }
    }
  }

  // Convolve grid with filter in Z direction
  for (int k = 0; k < grid->YResolution(); k++) {
    for (int j = 0; j < grid->XResolution(); j++) { 
      for (int i = 0; i < grid->ZResolution(); i++) 
        buffer[i] = grid->GridValue(j, k, i); 
      for (int i = 0; i < grid->ZResolution(); i++) { 
        RNScalar sum = 0.0;
        RNScalar weight = 0;
        RNScalar value = buffer[i];
        if (value != DONT_CARE) {
          sum += filter[0] * value;
          weight += filter[0];
        }
        int nsamples = i;
        if (nsamples > filter_radius) nsamples = filter_radius;
        for (int m = 1; m <= nsamples; m++) {
          RNScalar value = buffer[i - m];
          if (value != DONT_CARE) {
            sum += filter[m] * value;
            weight += filter[m];
          }
        }
        nsamples = grid->ZResolution() - 1 - i;
        if (nsamples > filter_radius) nsamples = filter_radius;
        for (int m = 1; m <= nsamples; m++) {
          RNScalar value = buffer[i + m];
          if (value != DONT_CARE) {
            sum += filter[m] * value;
            weight += filter[m];
          }
        }
        if (weight > 0) grid->SetGridValue(j, k, i, sum / weight);
      }
    }
  }

  // Deallocate memory
  delete [] filter;
  delete [] buffer;
}



static int
RasterizeAtoms(R3Grid *grid, const RNArray<PDBAtom *>& atoms, 
  int rasterization_type, RNBoolean grow, RNBoolean asa, RNBoolean proteinatoms, RNBoolean hetatoms, PDBElement *element)
{
  // Rasterize each atom according to the program arguments
  int atom_count = 0;
  for (int i = 0; i < atoms.NEntries(); i++) {
    PDBAtom *atom = atoms.Kth(i);
    PDBResidue *residue = atom->Residue();

    // Check atom
    if (!residue) continue;
    if (grow && (atom->NBonds() == 0)) continue;
    if (asa && (atom->accessible_surface_area == 0)) continue;
    if (!hetatoms && atom->IsHetAtom()) continue;
    if (!proteinatoms && !atom->IsHetAtom()) continue;
    if (element && (atom->Element() != element)) continue;
    atom_count++;

    // Determine splat amplitude
    RNScalar amplitude = 1;
    if (charge && (atom->Charge() != PDB_UNKNOWN)) {
      RNScalar charge_factor = pow(2.0, atom->Charge());
      amplitude *= charge_factor;
    }
    if (hydrophobicity && (atom->Hydrophobicity() != PDB_UNKNOWN)) {
      RNScalar hydrophobicity_factor = pow(2.0, atom->Hydrophobicity());
      amplitude *= hydrophobicity_factor;
    }
    if (accessible_surface_area && (atom->accessible_surface_area != PDB_UNKNOWN)) {
      RNScalar asa_factor = atom->accessible_surface_area / 10.0;
      amplitude *= asa_factor;
    }
    if (conservation && (residue->conservation != PDB_UNKNOWN)) {
      RNScalar conservation_factor = pow(2.0, 2.0 * residue->conservation - 1.0);
      amplitude *= conservation_factor;
    }

    // Splat atom
    if (rasterization_type == 0) grid->RasterizeWorldPoint(atom->Position(), amplitude); 
    else if (rasterization_type == 1) grid->RasterizeWorldSphere(atom->Position(), atom->Radius(), amplitude); 
    else if (rasterization_type == 2) grid->RasterizeWorldSphere(atom->Position(), atom->Radius(), amplitude); 
    else if (rasterization_type == 3) grid->RasterizeWorldSphere(atom->Position(), offset, amplitude); 
    else if (rasterization_type == 4) grid->RasterizeWorldSphere(atom->Position(), atom->Radius() + offset, amplitude, FALSE); 
  }

  // Run edge detector to compute surface from volume if rasterization_type=2
  if (rasterization_type == 2) {
    grid->Threshold(1.0E-6, 0, 1);
    grid->DetectEdges(); 
    grid->Threshold(1.0E-6, 0, 1);
  }

  // Blur grid
  if (sigma > 0) {
    if (ignore_zeros_when_blur) grid->Threshold(1.0E-20, DONT_CARE, R3_GRID_KEEP_VALUE);
    BlurGrid(grid, sigma);
    if (ignore_zeros_when_blur) grid->Threshold(DONT_CARE + 1.0E-20, 0, R3_GRID_KEEP_VALUE);
  }

  // Return number of atoms
  return atom_count;
}



static R3Grid *
CreateGrid(PDBFile *file, PDBResidue *ligand, PDBElement *element)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get array of atoms
  if (file->NModels() == 0) return NULL;
  PDBModel *model = file->Model(0);
  const RNArray<PDBAtom *> *atoms = (ligand) ? &(ligand->atoms) : &(model->atoms);

  // Compute world box
  R3Box world_box;
  if (world_center || (world_radius != 0)) {
    // Compute world centroid
    R3Point world_centroid;
    if (world_center) world_centroid = *world_center;
    else world_centroid = PDBCentroid(*atoms);

    // Compute world radius
    if (world_radius == 0) {
       world_radius = 3.0 * PDBAverageDistance(*atoms, world_centroid);
    }

    // Compute world box
    R3Point world_corner1(world_centroid - R3ones_vector * world_radius);
    R3Point world_corner2(world_centroid + R3ones_vector * world_radius);
    world_box = R3Box(world_corner1, world_corner2);
  }
  else {
    // Compute world box
    R3Box bbox = PDBBox(*atoms);
    R3Point world_corner1(bbox.Min() - R3ones_vector * world_border);
    R3Point world_corner2(bbox.Max() + R3ones_vector * world_border);
    world_box = R3Box(world_corner1, world_corner2);
  }

  // Compute grid resolution
  if (grid_spacing == 0) grid_spacing = 0.5;
  int xres = grid_resolution[0];
  if (xres == 0) {
    xres = (int) (world_box.XLength() / grid_spacing + 1);
    if (xres > max_grid_resolution) xres = max_grid_resolution;
  }
  int yres = grid_resolution[1];
  if (yres == 0) {
    yres = (int) (world_box.YLength() / grid_spacing + 1);
    if (yres > max_grid_resolution) yres = max_grid_resolution;
  }
  int zres = grid_resolution[2];
  if (zres == 0) {
    zres = (int) (world_box.ZLength() / grid_spacing + 1);
    if (zres > max_grid_resolution) zres = max_grid_resolution;
  }

  // Allocate grid
  R3Grid *grid = new R3Grid(xres, yres, zres);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    exit(-1);
  }

  // Set grid transformation
  grid->SetWorldToGridTransformation(world_box);

  // Rasterize atoms into grid
  int natoms = 0;
  RNBoolean grow = (grow_name) ? TRUE : FALSE;
  RNBoolean asa = (asa_name) ? TRUE : FALSE;
  if ((site_type == 1) && (ligand)) natoms = RasterizeAtoms(grid, ligand->atoms, rasterization_type, grow, asa, FALSE, TRUE, element);
  else if (site_type == 1) natoms = RasterizeAtoms(grid, model->atoms, rasterization_type, grow, asa, FALSE, TRUE, element);
  else natoms = RasterizeAtoms(grid, model->atoms, rasterization_type, grow, asa, TRUE, FALSE, element);

  // Threshold grid
  if (threshold != PDB_UNKNOWN) grid->Threshold(threshold, 0, 1);

  // Print statistics
  if (print_verbose) {
    printf("Created grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Grid Resolution = %d %d% d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    printf("  World Box = ( %g %g %g ) ( %g %g %g )\n", world_box[0][0], world_box[0][1], world_box[0][2], world_box[1][0], world_box[1][1], world_box[1][2]);
    printf("  World Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  World Volume = %g\n", grid->Volume());
    RNInterval range = grid->Range();
    printf("  Minimum = %g\n", range.Min());
    printf("  Maximum = %g\n", range.Max());
    printf("  L2Norm = %g\n", grid->L2Norm());
    printf("  # Atoms = %d\n", natoms);
    printf("  Sigma = %g\n", sigma);
    printf("  Offset = %g\n", offset);
    printf("  Conservation = %d\n", conservation);
    printf("  Accessible surface area = %d\n", accessible_surface_area);
    printf("  Hydrophobicity = %d\n", hydrophobicity);
    fflush(stdout);
  }

  // Return grid
  return grid;
}



static int 
NormalizeGrids(R3Grid **grids, int ngrids, int normalization_type)
{
  // Check arguments
  if (ngrids == 0) return 1;
  if (normalization_type == 0) return 1;

  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Check normalization type
  if ((normalization_type & 0x3) == 1) {
    // Normalize each grid 
    for (int i = 0; i < ngrids; i++) {
      grids[i]->Normalize();
    }
  }
  else if ((normalization_type & 0x3) == 2) {
    // Normalize each grid cell
    for (int x = 0; x < grids[0]->XResolution(); x++) {
      for (int y = 0; y < grids[0]->YResolution(); y++) {
        for (int z = 0; z < grids[0]->ZResolution(); z++) {
          // Compute sum of all grid entries at position
          RNScalar sum = 0;
          for (int i = 0; i < ngrids; i++) {
            sum += grids[i]->GridValue(x, y, z);
          }
  
          // Rescale so that they sum to one
          if (sum > 0) {
            for (int i = 0; i < ngrids; i++) {
               grids[i]->SetGridValue(x, y, z, grids[i]->GridValue(x, y, z) / sum);
            }
          }
        }
      }
    }
  }
  else if ((normalization_type & 0x3) == 3) {
    // Compute sum of all grids
    RNScalar sum = 0;
    for (int i = 0; i < ngrids; i++) {
      sum += grids[i]->L1Norm();
    }

    // Rescale so that they cumulatively sum to one
    if (sum > 0) {
      for (int i = 0; i < ngrids; i++) {
        grids[i]->Divide(sum);
      }
    }
  }
  else {
    RNAbort("Invalid normalization type");
  }

  // Print statistics
  if (print_verbose) {
    printf("Normalized grids ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Normalization type = %d\n", normalization_type);
    printf("  Norm by element ...\n");
    RNScalar L1_sum = 0;
    RNScalar L2_squared_sum = 0;
    for (int i = 0; i < ngrids; i++) { 
      RNScalar L1 = grids[i]->L1Norm(); L1_sum += L1;
      RNScalar L2_squared = grids[i]->L2NormSquared(); L2_squared_sum += L2_squared;
      printf("    Norms %s = %g %g\n", PDBelements[i].Name(), L1, sqrt(L2_squared));
    }
    printf("  L1Norm = %g\n", L1_sum);
    printf("  L2Norm = %g\n", sqrt(L2_squared_sum));
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int 
WriteGrid(R3Grid *grid, const char *grid_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write grid
  int status = grid->WriteFile(grid_name);

  // Print statistics
  if (print_verbose) {
    printf("Wrote grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Bytes = %d\n", (int) (status * sizeof(RNScalar)));
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
    printf("Usage: pdb2grd pdbfile gridfile [-resolution x y z] [-v]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) { 
        print_verbose = 1; 
      }
      else if (!strcmp(*argv, "-hetatoms")) { 
        site_type = 1; 
      }
      else if (!strcmp(*argv, "-ligand")) { 
        argc--; argv++; ligand_name = *argv; 
      }
      else if (!strcmp(*argv, "-asa")) { 
        argc--; argv++; asa_name = *argv; 
      }
      else if (!strcmp(*argv, "-consurf")) { 
        argc--; argv++; consurf_name = *argv; 
        conservation = TRUE;
      }
      else if (!strcmp(*argv, "-grow")) { 
        argc--; argv++; grow_name = *argv; 
      }
      else if (!strcmp(*argv, "-site")) { 
        argc--; argv++; site_type = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-rasterization")) { 
        argc--; argv++; rasterization_type = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-normalization")) { 
        argc--; argv++; normalization_type = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-radius")) { 
        argc--; argv++; world_radius = atof(*argv); 
      }
      else if (!strcmp(*argv, "-center")) { 
        world_center = new R3Point();
        argc--; argv++; (*world_center)[0] = atof(*argv); 
        argc--; argv++; (*world_center)[1] = atof(*argv); 
        argc--; argv++; (*world_center)[2] = atof(*argv); 
      }
      else if (!strcmp(*argv, "-resolution")) { 
        argc--; argv++; grid_resolution[0] = atoi(*argv); 
        argc--; argv++; grid_resolution[1] = atoi(*argv); 
        argc--; argv++; grid_resolution[2] = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-max_resolution")) { 
        argc--; argv++; max_grid_resolution = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-spacing")) { 
        argc--; argv++; grid_spacing = atof(*argv); 
      }
      else if (!strcmp(*argv, "-border")) { 
        argc--; argv++; world_border = atof(*argv); 
      }
      else if (!strcmp(*argv, "-sigma")) { 
        argc--; argv++; sigma = atof(*argv); 
      }
      else if (!strcmp(*argv, "-threshold")) { 
        argc--; argv++; threshold = atof(*argv); 
      }
      else if (!strcmp(*argv, "-offset")) { 
        argc--; argv++; offset = atof(*argv); 
      }
      else if (!strcmp(*argv, "-charge")) { 
        charge = TRUE; 
      }
      else if (!strcmp(*argv, "-hydrophobicity")) { 
        hydrophobicity = TRUE; 
      }
      else if (!strcmp(*argv, "-accessible_surface_area")) { 
        accessible_surface_area = TRUE;
      }
      else if (!strcmp(*argv, "-elements")) { 
        elements = TRUE; 
      }
      else if (!strcmp(*argv, "-ignore_zeros_when_blur")) { 
        ignore_zeros_when_blur = TRUE; 
      }
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        return 0;
      }
    }
    else {
      if (!pdb_name) pdb_name = *argv;
      else if (!grid_name) grid_name = *argv;
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        return 0;
      }
    }
    argv++; argc--;
  }

  // Check pdb filename
  if (!pdb_name) {
    fprintf(stderr, "You did not specify a pdb file.\n");
    return 0;
  }

  // Check grid filename
  if (!grid_name) {
    fprintf(stderr, "You did not specify a grid file.\n");
    return 0;
  }

  // Check grid resolution
  if ((grid_resolution[0] < 0) || (grid_resolution[1] < 0) || (grid_resolution[2] < 0)) {
    fprintf(stderr, "Invalid grid resolution: %d %d %d\n", grid_resolution[0], grid_resolution[1], grid_resolution[2]);
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
  PDBFile *file = ReadPDB(pdb_name);
  if (!file) exit(-1);

  // Find ligand
  PDBResidue *ligand = NULL;
  if (ligand_name) {
    ligand = FindLigand(file, ligand_name);
    if (!ligand) exit(-1);
  }

  // Read accessible surface area file
  if (asa_name) {
    int status = ReadASAFile(file, asa_name);
    if (!status) exit(-1);
  }

  // Read grow file
  if (grow_name) {
    int status = ReadGrowFile(file, grow_name);
    if (!status) exit(-1);
  }

  // Read consurf files
  if (consurf_name) {
    int status = ReadConsurfFiles(file, consurf_name);
    if (!status) exit(-1);
  }

  // Check if should make separate grid for each element
  if (elements) {
    R3Grid *grids[5];

    // Create grids (1=C, 2=N, 3=O, 4=P, 5=S)
    for (int i = 0; i < 5; i++) {
      grids[i] = CreateGrid(file, ligand, &PDBelements[i+1]);
      if (!grids[i]) exit(-1);
    }

    // Normalize grids
    if (!NormalizeGrids(grids, 5, normalization_type)) exit(-1);

    // Write grids
    for (int i = 0; i < 5; i++) {
      char name[256];
      strcpy(name, grid_name);
      char *namep = strstr(name, ".grd");
      if (!namep) namep = &name[strlen(name)];
      sprintf(namep, "-%s.grd", PDBelements[i+1].Name());
      int status = WriteGrid(grids[i], name);
      if (!status) exit(-1);
    }
  }
  else {
    // Create grid 
    R3Grid *grid = CreateGrid(file, ligand, NULL);
    if (!grid) exit(-1);

    // Normalize grid
    if (!NormalizeGrids(&grid, 1, normalization_type)) exit(-1);

    // Write grid
    int status = WriteGrid(grid, grid_name);
    if (!status) exit(-1);
  }

  // Return success
  return 0;
}
