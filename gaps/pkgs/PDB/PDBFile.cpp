// Source file for PDBFile class



// Include files

#include "PDB.h"
 


PDBFile::
PDBFile(const char *name)
  : mark(0),
    value(0),
    data(NULL)
{
  // Assign ID
  static int PDBnext_file_id = 0;
  id = PDBnext_file_id++;

  // Copy name
  if (name) { strncpy(this->name, name, 128); this->name[127] = 0; }
  else this->name[0] = '\0';
}



PDBFile::
~PDBFile(void)
{
  // Free headers and trailers
  for (int i = 0; i < headers.NEntries(); i++) 
    free(headers[i]);
  for (int i = 0; i < trailers.NEntries(); i++) 
    free(trailers[i]);

  // Delete models
  while (NModels()) delete models.Tail();
}



const RNRgb& PDBFile::
Color(void) const
{
  static const RNRgb colors[8] = { 
    RNRgb(0, 0, 1), RNRgb(1, 0, 0), RNRgb(0, 0.7, 0), RNRgb(0, 1, 1), 
    RNRgb(1, 0, 1), RNRgb(1, 0.5, 1), RNRgb(1, 1, 0.5), RNRgb(0.5, 1, 1)
  };
  return colors[id % 8];
}



const R3Box PDBFile::
BBox(void) const
{
  // Return bounding box (including atom radii)
  R3Box bbox = R3null_box;
  for (int i = 0; i < NModels(); i++)
    bbox.Union(Model(i)->BBox());
  return bbox;
}



R3Point PDBFile::
Centroid(void) const
{
  // Return centroid of all atoms
  int atom_count = 0;
  R3Point centroid(0, 0, 0);
  for (int i = 0; i < NModels(); i++) {
    PDBModel *model = Model(i);
    int natoms = model->NAtoms();
    centroid += natoms * model->Centroid();
    atom_count += natoms;
  }
  centroid /= atom_count;
  return centroid;
}



RNLength PDBFile::
Radius(void) const
{
  // Compute centroid
  R3Point centroid = Centroid();

  // Compute maximum distance to centroid
  RNLength max_distance = 0;
  for (int i = 0; i < NModels(); i++) {
    PDBModel *model = Model(i);
    RNLength distance = PDBMaxDistance(model->atoms, centroid);
    if (distance > max_distance) max_distance = distance;
  }

  // Return maximum distance found
  return max_distance;
}



void PDBFile::
Transform(const R3Affine& affine)
{
  // Transform all models by affine transformation
  for (int i = 0; i < models.NEntries(); i++) 
    models[i]->Transform(affine);
}



PDBModel *PDBFile::
InsertCopy(PDBModel *m)
{
  // Insert copy of model into this model
  PDBModel *model =  new PDBModel(this, m->Name());
  assert(model);

  // Insert copy of all m's chains
  for (int i = 0; i < m->NChains(); i++) 
    model->InsertCopy(m->Chain(i));

  // Return model
  return model;
}



PDBStructureType PDBFile::
FindAny(const char *str, PDBModel **m, PDBChain **c, PDBResidue **r, PDBAtom **a, PDBStructureType maxlevel) const
{
  // Parse first token
  if (!str) return PDB_FILE;
  if (strlen(str) < 1) return PDB_FILE;
  char *strp = (char *) str;
  while (*strp) { if (*strp == '-') break; strp++; }

  // Check if PDB code is at beginning of string
  if (((strp - str) == 4) && (*strp == '-')) 
    return FindAny(&str[5], m, c, r, a, maxlevel);

  // Temporarily modify string
  char save = *strp;
  *strp = '\0';

  // Find model from string
  PDBModel *result = NULL;
  for (int i = 0; i < NModels(); i++) {
    PDBModel *model = Model(i);
    if (!strcmp(model->Name(), str)) {
      result = model;
      break;
    }
  }

  // Restore string
  *strp = save;

  // Fill return values and consider matches at deeper levels
  PDBStructureType level = PDB_FILE;
  if (result) {
    level = PDB_MODEL;
    if (m) *m = result;
    if (strp && *strp && (maxlevel > level)) {
      level = result->FindAny(strp+1, c, r, a, maxlevel);
    }
  }

  // Return level of match 
  return level;
}



int PDBFile::
ReadFile(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    RNFail("Unable to open PDB file: %s", filename);
    return 0;
  }

  // Read file
  int status = Read(fp);

  // Close file
  fclose(fp);

  // Remember filename 
  const char *filenamep = strchr(filename, '/');
  if (filenamep) filenamep += 1;
  else filenamep = filename;
  int filename_length = strlen(filenamep);
  if (filename_length > 127) filename_length = 127;
  strncpy(this->name, filenamep, 128);
  if (!strcmp(&filenamep[filename_length-4], ".pdb")) 
    this->name[filename_length-4] = '\0';

  // Return number of models read
  return status;
}



int PDBFile::
WriteFile(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    RNFail("Unable to open PDB file: %s", filename);
    return 0;
  }

  // Write file
  int status = Write(fp);

  // Close file
  fclose(fp);

  // Return number of models written
  return status;
}



static char *
PDBToken(char *buffer, int start, int stop, int length)
{
  static char token[128];
  assert(stop-start+1 < 32);
  char *tokenp = token;
  if (stop > length) stop = length;
  for (int i = start-1; i <= stop-1; i++) 
    if (!isspace(buffer[i])) 
      *(tokenp++) = buffer[i];
  *tokenp = '\0';
  return token;
}



int PDBFile::
Read(FILE *fp)
{
  // Create a model
  char model_name[16];
  sprintf(model_name, "%d", NModels()+1);
  PDBModel *model = new PDBModel(this, model_name);
  assert(model);

  // Read lines from PDB file
  int atom_count = 0;
  char buffer[1024];
  while (fgets(buffer, 1024, fp)) {
    // Compute buffer length
    int len = strlen(buffer);

    // Check record type
    if ((!strncmp(&buffer[0], "ATOM  ", 6)) || (!strncmp(&buffer[0], "HETATM  ", 6))) {
      // Parse atom fields 
      RNBoolean hetatm = (buffer[0] == 'H');
      int serial = atoi(PDBToken(buffer, 7, 11, len));
      char atomName[8]; strncpy(atomName, &buffer[12], 4); atomName[4] = '\0';
      int altLoc = buffer[16];
      char resName[4]; strncpy(resName, PDBToken(buffer, 18, 20, len), 4); 
      char chainName[4]; chainName[0] = buffer[21]; chainName[1] = '\0';
      int resSeq = atoi(PDBToken(buffer, 23, 26, len));
      int iCode = buffer[26];
      RNScalar x = atof(PDBToken(buffer, 31, 38, len));
      RNScalar y = atof(PDBToken(buffer, 39, 46, len));
      RNScalar z  = atof(PDBToken(buffer, 47, 54, len));
      char occupancy_string[8]; strncpy(occupancy_string, PDBToken(buffer, 55, 60, len), 8); 
      RNScalar occupancy = (strlen(occupancy_string) > 0) ? atof(occupancy_string) : 1;
      char tempFactor_string[8]; strncpy(tempFactor_string, PDBToken(buffer, 61, 66, len), 8); 
      RNScalar tempFactor = (strlen(tempFactor_string) > 0) ? atof(tempFactor_string) : 0;
      char elementName[4]; strncpy(elementName, PDBToken(buffer, 77, 78, len), 4); 
      char charge_string[4]; strncpy(charge_string, PDBToken(buffer, 79, 80, len), 4);
      RNScalar charge = (strlen(charge_string) > 0) ? atof(charge_string) : PDB_UNKNOWN;

      // Check for water atoms
      if (!strcmp(resName, "HOH")) {
        // fprintf(stderr, "Skipping atom %d in water molecule\n", serial);
        continue;
      }

      // Get element
      PDBElement *element = NULL;
      if (elementName[0]) element = PDBFindElement(elementName);
      if (!element && atomName[0] ){
        for (int i = 0; i < PDBnelements; i++) {
          if (!strncmp(atomName, PDBelements[i].Name(), 2)) {
            element = &PDBelements[i];
            break;
          }
        }
      }
      if (!element) {
        for (int i = 0; i < PDBnelements; i++) {
          if (!strncmp(&atomName[1], PDBelements[i].Name(), strlen(PDBelements[i].Name()))) {
            element = &PDBelements[i];
            break;
          }
        }
      }

      // Check for hydrogen atoms
      if (element && (element->ID() == PDB_H_ELEMENT)) { 
        // fprintf(stderr, "Skipping hydrogen atom %d (%s)\n", serial, elementName);
        continue;
      }

      // Check again for hydrogen atoms
      if (atomName[1] == 'H') {
        // fprintf(stderr, "Skipping hydrogen atom %d (%s)\n", serial, elementName);
        continue;
      }

      // Get/create chain
      PDBChain *chain = model->FindChain(chainName);
      if (!chain) {
        chain = new PDBChain(model, chainName);
        assert(chain);
      }

      // Get/create residue
      PDBResidue *residue = chain->FindResidue(resName, resSeq, iCode);
      if (!residue) {
        residue = new PDBResidue(model, chain, resName, resSeq, iCode);
        assert(residue);
      }

      // Get charge from corresponding atom in amino acid 
      if (charge == PDB_UNKNOWN) {
        // Get charge from Hugh's simplified table
        if (!strcmp(resName, "LYS") && !strcmp(atomName, " N2 ")) charge = 1;
        else if (!strcmp(resName, "ARG") && (!strcmp(atomName, " NH1") || !strcmp(atomName, " NH2"))) charge = 0.5;
        else if (!strcmp(resName, "GLU") && (!strcmp(atomName, " OE1") || !strcmp(atomName, " OE2"))) charge = -0.5;
        else if (!strcmp(resName, "ASP") && (!strcmp(atomName, " OD1") || !strcmp(atomName, " OD2"))) charge = -0.5;
        else if (!strcmp(atomName, "OXT")) charge = -1;
        else if (element) charge = element->Charge();
      }

      // Check altLoc
      if (altLoc != ' ') {
        if (residue->FindAtom(atomName)) {
          // Print warning
          // fprintf(stderr, "Skipping atom %d with non-zero altLoc\n", serial);
          continue;
        }
      }

      // Create atom
      PDBAtom *atom = new PDBAtom(model, chain, residue, element,
        serial, atomName, altLoc, x, y, z, occupancy, tempFactor, charge, hetatm);
      if (!atom) {
        fprintf(stderr, "Unable to create atom: %s\n", atomName);
        abort();
      }

      // Count atoms
      atom_count++;
    }
    else if (!strncmp(&buffer[0], "ENDMDL", 6)) {
      if (model->NAtoms() > 0) {
        // Create another model
        char model_name[16];
        sprintf(model_name, "%d", NModels()+1);
        model = new PDBModel(this, model_name);
        assert(model);
      }
    }
    else {
      // Save irrelevant line so can write them out later
      if (atom_count == 0) headers.Insert(strdup(buffer));
      else trailers.Insert(strdup(buffer));
    }
  }

  // Remove last model, if it was empty
  if (model->NAtoms() == 0) delete model; 

  // Return number of models read
  return NModels();
}



int PDBFile::
Write(FILE *fp) const
{
  // Write all header lines
  for (int i = 0; i < headers.NEntries(); i++) {
    fprintf(fp, "%s\n", headers.Kth(i));
  }

  // Write all models
  for (int i = 0; i < NModels(); i++) {
    // Get Model
    PDBModel *model = Model(i);

    // Write model delimiter
    if (NModels() > 1) fprintf(fp, "%-6s%8d\n", "MODEL", i);

    // Write all atoms of model
    for (int i = 0; i < model->NAtoms(); i++) {
      // Get atom
      PDBAtom *atom = model->Atom(i);

      // Get element
      PDBElement *element = atom->Element();

      // Get residue
      PDBResidue *residue = atom->Residue();
      if (!residue) { 
        fprintf(stderr, "Null residue for atom %d\n", atom->Serial()); 
        continue; 
      }

      // Get chain
      PDBChain *chain = residue->Chain();
      if (!chain) {
        fprintf(stderr, "Null chain for atom %d\n", atom->Serial());
        continue;
      }

      // Print line to PDB file
      fprintf(fp, "%-6s%5d %-4s%c%3s %1s%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n",
              (atom->IsHetAtom()) ? "HETATM" : "ATOM",
              atom->Serial(), atom->Name(), atom->AlternateLocation(),
              residue->Name(), chain->Name(), residue->Sequence(), residue->InsertionCode(), 
              atom->Position().X(), atom->Position().Y(), atom->Position().Z(),
              atom->Occupancy(), atom->TempFactor(), (element) ? element->Name() : "");
    }

    // Write model delimiter
    if (NModels() > 1) fprintf(fp, "%-6s\n", "ENDMDL");
  } 
    
  // Write all trailing lines
  for (int i = 0; i < trailers.NEntries(); i++) {
    fprintf(fp, "%s\n", trailers.Kth(i));
  }

  // Return number of models written
  return NModels();
}



PDBFile *PDBFile::
CopyBiomolecule(void)
{
  char buffer[1024], *bufferp;
  char buf1[128], buf2[128], buf3[128];

  // Get model
  if (NModels() == 0) return NULL;
  PDBModel *model = Model(0);

  // Create new file
  PDBFile *file2 = new PDBFile(Name());
  assert(file2);
  
  // Create new model
  PDBModel *model2 = new PDBModel(file2, "1");
  assert(model2);

  // Process remark 350 
  RNArray<PDBChain *> chains;
  R3Vector translation = R3zero_vector;
  R4Matrix rotation = R4identity_matrix;
  RNBoolean found_biomt = FALSE;
  for (int i = 0; i < headers.NEntries(); i++) {
    const char *remark = headers.Kth(i);
    if (sscanf(remark, "%s%s%s", buf1, buf2, buf3) == 3) {
      if (!strcmp(buf1, "REMARK")) {
        if (!strcmp(buf2, "350")) {
          if (!strcmp(buf3, "APPLY")) {
            // Parse list of chains
            strncpy(buffer, remark, 1024); buffer[1023] = '\0';
            bufferp = strtok(buffer, " \t");
            if (!bufferp || strcmp(bufferp, "REMARK")) { fprintf(stderr, "Error 1 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "350")) { fprintf(stderr, "Error 2 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "APPLY")) { fprintf(stderr, "Error 3 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "THE")) { fprintf(stderr, "Error 4 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "FOLLOWING")) { fprintf(stderr, "Error 5 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "TO")) { fprintf(stderr, "Error 6 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "CHAINS:")) { fprintf(stderr, "Error 7 parsing remark 350\n"); return NULL; }
            if (found_biomt) { translation = R3zero_vector; rotation = R4identity_matrix; chains.Empty(); found_biomt = FALSE; }
            while (TRUE) {
              // Read chain names
              bufferp = strtok(NULL, " \t\n");
              if (!bufferp) break;
              char chain_name[16];
              if (!strcmp(bufferp, "NULL")) strcpy(chain_name, " ");
              else strcpy(chain_name, bufferp);
              PDBChain *chain = model->FindChain(chain_name);
              if (!chain) { fprintf(stderr, "Unable to find chain %s found in remark 350\n", bufferp); return NULL; }
              chains.Insert(chain);
            }
          }
          else if (!strncmp(buf3, "BIOMT", 5)) {
            // Parse transformation matrix
            found_biomt = TRUE;
            char buffer[1024], *bufferp;
            strncpy(buffer, remark, 1024); buffer[1023] = '\0';
            bufferp = strtok(buffer, " \t");
            if (!bufferp || strcmp(bufferp, "REMARK")) { fprintf(stderr, "Error 8 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "350")) { fprintf(stderr, "Error 9 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strncmp(bufferp, "BIOMT", 5)) { fprintf(stderr, "Error 10 parsing remark 350\n"); return NULL; }
            int row_number = atoi(&bufferp[strlen(bufferp)-1]);
            if ((row_number < 1) || (row_number > 3))  { fprintf(stderr, "Error 11 parsing remark 350\n"); return NULL; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 12 parsing remark 350\n"); return NULL; }
            // int matrix_number = atoi(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 13 parsing remark 350\n"); return NULL; }
            rotation[row_number-1][0] = atof(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 14 parsing remark 350\n"); return NULL; }
            rotation[row_number-1][1] = atof(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 15 parsing remark 350\n"); return NULL; }
            rotation[row_number-1][2] = atof(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 16 parsing remark 350\n"); return NULL; }
            translation[row_number-1] = atof(bufferp);

            // Copy chain and apply transformation matrix
            if (row_number == 3) {
              // Compute transformation
              R3Affine affine(R3identity_affine);
              affine.Translate(translation);
              affine.Transform(R3Affine(rotation));

              // Apply transformation to chains
              for (int j = 0; j < chains.NEntries(); j++) {
                PDBChain *chain = chains.Kth(j);

                // Get name for chain
                char name2[4] = { 0, 0, 0, 0 };
                assert(strlen(chain->Name()) == 1);
                if (!model2->FindChain(chain->Name())) {
                  // Use original name
                  strcpy(name2, chain->Name());
                }
                else {
                  // Search for an available name
                  name2[1] = '\0';
                  for (name2[0] = 'A'; name2[0] <= 'z'; name2[0]++) {
                    if (!model->FindChain(name2) && !model2->FindChain(name2)) {
                      break;
                    }
                  }
                }

                // Insert chain
                PDBChain *chain2 = model2->InsertCopy(chain);
                strcpy(chain2->name, name2);

                // Transform chain
                if (!affine.IsIdentity()) {
                  chain2->Transform(affine);
                }
              }
            }
          }
        }
      }
    }
  }

  // Copy header stuff, fixing REMARK 350
  RNBoolean first = TRUE;
  for (int i = 0; i < headers.NEntries(); i++) {
    const char *remark = headers.Kth(i);
    if ((sscanf(remark, "%s%s%s", buf1, buf2, buf3) == 3) && 
        (!strcmp(buf1, "REMARK") && !strcmp(buf2, "350") && 
         (!strcmp(buf3, "APPLY") || !strncmp(buf3, "BIOMT", 5)))) {
      if (first) {
        // Only add REMARK 350 stuff once
        first = FALSE;

        // Insert a new REMARK 350 APPLY line
        strcpy(buffer, "REMARK 350 APPLY THE FOLLOWING TO CHAINS: ");
        for (int j = 0; j < model2->NChains(); j++) { 
          if (j > 0) strcat(buffer, ", "); 
          strcat(buffer, model2->Chain(j)->Name()); 
        }
        strcat(buffer, "\n");
        file2->headers.Insert(strdup(buffer));

        // Insert new REMARK 350 BIOMT lines
        file2->headers.Insert(strdup("REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n"));
        file2->headers.Insert(strdup("REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n"));
        file2->headers.Insert(strdup("REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n"));
      }
    }
    else {
      // Copy header line
      file2->headers.Insert(strdup(headers[i]));
    }
  }

  // Copy trailer stuff
  for (int i = 0; i < trailers.NEntries(); i++) 
    file2->trailers.Insert(strdup(trailers[i]));

  // Return new file
  return file2;
}



int PDBFile::
IsBiomolecule(void) const
{
  char buffer[1024], *bufferp;
  char buf1[128], buf2[128], buf3[128];

  // Get model
  if (NModels() == 0) return 0;
  PDBModel *model = Model(0);

  // Process remark 350 
  PDBClearMarks();
  RNArray<PDBChain *> chains;
  R3Vector translation = R3zero_vector;
  R4Matrix rotation = R4identity_matrix;
  RNBoolean found_remark350 = FALSE;
  RNBoolean found_biomt = FALSE;
  for (int i = 0; i < headers.NEntries(); i++) {
    const char *remark = headers.Kth(i);
    if (sscanf(remark, "%s%s%s", buf1, buf2, buf3) == 3) {
      if (!strcmp(buf1, "REMARK")) {
        if (!strcmp(buf2, "350")) {
          found_remark350 = TRUE;
          if (!strcmp(buf3, "APPLY")) {
            // Parse list of chains
            strncpy(buffer, remark, 1024); buffer[1023] = '\0';
            bufferp = strtok(buffer, " \t");
            if (!bufferp || strcmp(bufferp, "REMARK")) { fprintf(stderr, "Error 1 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "350")) { fprintf(stderr, "Error 2 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "APPLY")) { fprintf(stderr, "Error 3 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "THE")) { fprintf(stderr, "Error 4 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "FOLLOWING")) { fprintf(stderr, "Error 5 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "TO")) { fprintf(stderr, "Error 6 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "CHAINS:")) { fprintf(stderr, "Error 7 parsing remark 350\n"); return 0; }
            if (found_biomt) { translation = R3zero_vector; rotation = R4identity_matrix; chains.Empty(); found_biomt = FALSE; }
            while (TRUE) {
              // Read chain names
              bufferp = strtok(NULL, " \t\n");
              if (!bufferp) break;
              char chain_name[16];
              if (!strcmp(bufferp, "NULL")) strcpy(chain_name, " ");
              else strcpy(chain_name, bufferp);
              PDBChain *chain = model->FindChain(chain_name);
              if (!chain) { fprintf(stderr, "Unable to find chain %s found in remark 350\n", bufferp); return 0; }

              // Mark chain
              chain->SetMark();
            }
          }
          else if (!strncmp(buf3, "BIOMT", 5)) {
            // Parse transformation matrix
            found_biomt = TRUE;
            char buffer[1024], *bufferp;
            strncpy(buffer, remark, 1024); buffer[1023] = '\0';
            bufferp = strtok(buffer, " \t");
            if (!bufferp || strcmp(bufferp, "REMARK")) { fprintf(stderr, "Error 8 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strcmp(bufferp, "350")) { fprintf(stderr, "Error 9 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || strncmp(bufferp, "BIOMT", 5)) { fprintf(stderr, "Error 10 parsing remark 350\n"); return 0; }
            int row_number = atoi(&bufferp[strlen(bufferp)-1]);
            if ((row_number < 1) || (row_number > 3))  { fprintf(stderr, "Error 11 parsing remark 350\n"); return 0; }
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 12 parsing remark 350\n"); return 0; }
            // int matrix_number = atoi(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 13 parsing remark 350\n"); return 0; }
            rotation[row_number-1][0] = atof(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 14 parsing remark 350\n"); return 0; }
            rotation[row_number-1][1] = atof(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 15 parsing remark 350\n"); return 0; }
            rotation[row_number-1][2] = atof(bufferp);
            bufferp = strtok(NULL, " \t");
            if (!bufferp || isalpha(*bufferp)) { fprintf(stderr, "Error 16 parsing remark 350\n"); return 0; }
            translation[row_number-1] = atof(bufferp);

            // Check transformation matrix
            if (row_number == 3) {
              // Check transformation
              if (!rotation.IsIdentity()) return 0;
              if (!translation.IsZero()) return 0;
            }
          }
        }
      }
    }
  }


  // Check if found remark 350
  if (found_remark350) {
    // Check if all chains are marked
    for (int i = 0; i < model->NChains(); i++) {
      PDBChain *chain = model->Chain(i);

      // Check if chain isn't all hetatoms
      RNBoolean has_atoms = FALSE;
      for (int j = 0; j < chain->NAtoms(); j++) {
        PDBAtom *atom = chain->Atom(j);
        if (!atom->IsHetAtom()) has_atoms = TRUE;
      }

      // Check if chain is marked
      if (has_atoms && !model->Chain(i)->IsMarked()) {
        return 0;
      } 
    }
  }
  else {
    // File has no remark 350 -- don't know if is biomolecule
    return 2;
  }

  // Passed all tests -- is biomolecule
  return 1;
}



static int 
ReadConsurfFile(PDBChain *chain, const char *filename)
{
  // Initialize all conservation scores
  for (int i = 0; i < chain->NResidues(); i++) {
    PDBResidue *residue = chain->Residue(i);
    residue->conservation = PDB_UNKNOWN;
  }

  // Open conservation file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    // It is OK for the file to be missing
    // fprintf(stderr, "Unable to open conservation file %s\n", filename);
    return -1;
  }

  // Read lines from conservation file
  int residue_count = 0;
  int line_count = 0;
  char buffer[1024];
  while (fgets(buffer, 1024, fp)) {
    // Increment line count
    line_count++;

    // Check for comment or blank line
    char *bufferp = buffer;
    while (isspace(*bufferp)) bufferp++;
    if (*bufferp == '\0') continue;
    if (!isdigit(*bufferp)) continue;

    // Parse residue serial
    bufferp = strtok(bufferp, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse line %d of %s\n", line_count, filename); return 0; }
    // int residue_serial = atoi(bufferp);

    // Parse residue letter
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse residue letter at line %d of %s\n", line_count, filename); return 0; }
    // char residue_letter = *bufferp;

    // Parse residue string (NameSeq:Chain)
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse residue string at line %d of %s\n", line_count, filename); return 0; }
    char residue_string[64];
    strcpy(residue_string, bufferp);
    char *residue_stringp = strchr(residue_string, ':');
    if (residue_stringp) { *residue_stringp = '\0'; }
    // const char *residue_chain = (residue_stringp) ? residue_stringp + 1 : " ";
    char *residue_code = residue_string;
    int residue_sequence = atoi(&residue_string[3]);
    residue_string[3] = '\0';

    // Parse residue conservation score
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse score at line %d of %s\n", line_count, filename); return 0; }
    // RNScalar residue_conservation_score = atof(bufferp);

    // Parse residue conservation color
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse color at line %d of %s\n", line_count, filename); return 0; }
    int residue_conservation_color = atoi(bufferp);

    // Parse residue conservation number of homologues
    bufferp = strtok(NULL, "/");
    if (!bufferp) { fprintf(stderr, "Unable to parse number of homologues at line %d of %s\n", line_count, filename); return 0; }
    // int residue_conservation_num_hom = atoi(bufferp);

    // Parse residue conservation total number of homologues
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse total homologues at line %d of %s\n", line_count, filename); return 0; }
    // int residue_conservation_total_hom = atoi(bufferp);

    // Parse residue variety
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse residue variety at line %d of %s\n", line_count, filename); return 0; }
    // char *residue_variety = bufferp;

    // Find residue in chain
    PDBResidue *residue = chain->FindResidue(residue_code, residue_sequence, ' ');
    if (!residue) {
      fprintf(stderr, "Residue at line %d of %s not found in PDB file\n", line_count, filename); 
      continue;
    }

    // Assign residue conservation
    residue->conservation = (RNScalar) residue_conservation_color / 9.0;

    // Increment residue counter
    residue_count++;
  }

  // Close conservation file
  fclose(fp);

  // Return number of residues considered
  return residue_count;
}



int PDBFile::
ReadConsurfFiles(const char *consurf_basename)
{
  // Check number of models
  if (NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", consurf_basename);
    return 0;
  }

  // Read conservations for all chains
  int residue_count = 0;
  char chain_filename[256];
  PDBModel *model = Model(0);
  for (int i = 0; i < model->NChains(); i++) {
    PDBChain *chain = model->Chain(i);
    char chain_name[4] = { 0 };
    strcpy(chain_name, chain->Name());
    if (chain_name[0] == ' ') chain_name[0] = '_';
    strcpy(chain_filename, consurf_basename);
    strcat(chain_filename, chain_name);
    strcat(chain_filename, ".gradesPE");
    int status = ReadConsurfFile(chain, chain_filename);
    if (status == 0) return 0;
    else if (status > 0) {
      residue_count += status;
    }
  }

  // Return number of residues considered
  return residue_count;
}



static int 
ReadHsspFile(PDBChain *chain, const char *filename)
{
  // Initialize all conservation scores
  for (int i = 0; i < chain->NResidues(); i++) {
    PDBResidue *residue = chain->Residue(i);
    residue->conservation = PDB_UNKNOWN;
  }

  // Open conservation file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    // It is OK for the file to be missing
    // fprintf(stderr, "Unable to open conservation file %s\n", filename);
    return -1;
  }

  // Read lines from conservation file
  int residue_count = 0;
  int line_count = 0;
  char buffer[1024];
  while (fgets(buffer, 1024, fp)) {
    // Increment line count
    line_count++;

    // Check for comment or blank line
    char *bufferp = buffer;
    while (isspace(*bufferp)) bufferp++;
    if (*bufferp == '\0') continue;
    if (*bufferp == '#') continue;

    // Parse residue serial
    bufferp = strtok(bufferp, " ");
    if (!bufferp) { fprintf(stderr, "Unable to parse line %d of %s\n", line_count, filename); return 0; }
    // int residue_serial = atoi(bufferp);

    // Parse residue letter
    bufferp = strtok(NULL, " ");
    if (!bufferp) { fprintf(stderr, "Unable to parse residue letter at line %d of %s\n", line_count, filename); return 0; }
    char residue_letter = *bufferp;

    // Parse residue conservation mean
    bufferp = strtok(NULL, "[");
    if (!bufferp) { fprintf(stderr, "Unable to parse mean at line %d of %s\n", line_count, filename); return 0; }
    RNScalar residue_conservation_mean = atof(bufferp);

    // Parse residue conservation minQQ
    bufferp = strtok(NULL, ",");
    if (!bufferp) { fprintf(stderr, "Unable to parse minQQ at line %d of %s\n", line_count, filename); return 0; }
    // RNScalar residue_conservation_minQQ = atof(bufferp);

    // Parse residue conservation maxQQ
    bufferp = strtok(NULL, "]");
    if (!bufferp) { fprintf(stderr, "Unable to parse maxQQ at line %d of %s\n", line_count, filename); return 0; }
    // RNScalar residue_conservation_maxQQ = atof(bufferp);

    // Parse residue conservation std
    bufferp = strtok(NULL, " ");
    if (!bufferp) { fprintf(stderr, "Unable to parse std at line %d of %s\n", line_count, filename); return 0; }
    // RNScalar residue_conservation_std = atof(bufferp);

    // Parse residue conservation number of homologues
    bufferp = strtok(NULL, "/");
    if (!bufferp) { fprintf(stderr, "Unable to parse number of homologues at line %d of %s\n", line_count, filename); return 0; }
    // int residue_conservation_num_hom = atoi(bufferp);

    // Parse residue conservation total number of homologues
    bufferp = strtok(NULL, "/");
    if (!bufferp) { fprintf(stderr, "Unable to parse total homologues at line %d of %s\n", line_count, filename); return 0; }
    // int residue_conservation_total_hom = atoi(bufferp);

    // Get residue from PDB file
    PDBResidue *residue = NULL;
    PDBAminoAcid *aminoacid = NULL;
    while ((!residue || !aminoacid) && (residue_count < chain->NResidues())) {
      residue = chain->Residue(residue_count++);
      aminoacid = residue->AminoAcid();
    }

    // Check residue and aminoacid
    if (!residue || !aminoacid) {
      fprintf(stderr, "Residue at line %d of %s not found in PDB file\n", line_count, filename); 
      break;
    }

    // Check if amino acid letters match -- if not, assume that extra residue is in hsspfile
    if (aminoacid && (aminoacid->Letter() != residue_letter)) {
      fprintf(stderr, "Skipping residue %s %s %d (%c) in %s\n", 
        chain->Name(), residue->Name(), residue->Sequence(), aminoacid->Letter(), filename);
      residue_count--;
      continue;
    }

    // Assign residue conservation
    residue->conservation =  1 - residue_conservation_mean;
  }

  // Close conservation file
  fclose(fp);

  // Return number of residues considered
  return residue_count;
}



int PDBFile::
ReadHsspFiles(const char *hssp_basename)
{
  // Check number of models
  if (NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", hssp_basename);
    return 0;
  }

  // Read conservations for all chains
  int residue_count = 0;
  char chain_filename[256];
  PDBModel *model = Model(0);
  for (int i = 0; i < model->NChains(); i++) {
    PDBChain *chain = model->Chain(i);
    char chain_name[4] = { 0 };
    strcpy(chain_name, chain->Name());
    if (chain_name[0] == ' ') chain_name[0] = '_';
    strcpy(chain_filename, hssp_basename);
    strcat(chain_filename, chain_name);
    strcat(chain_filename, ".hssp.orig");
    int status = ReadHsspFile(chain, chain_filename);
    if (status == 0) return 0;
    else if (status > 0) {
      residue_count += status;
    }
  }

  // Return number of residues considered
  return residue_count;
}



static int 
ReadJsdFile(PDBChain *chain, const char *filename)
{
  // Initialize all conservation scores
  for (int i = 0; i < chain->NResidues(); i++) {
    PDBResidue *residue = chain->Residue(i);
    residue->conservation = PDB_UNKNOWN;
  }

  // Open conservation file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    // It is OK for the file to be missing
    // fprintf(stderr, "Unable to open conservation file %s\n", filename);
    return -1;
  }

  // Read lines from conservation file
  int residue_count = 0;
  int line_count = 0;
  char buffer[1024];
  while (fgets(buffer, 1024, fp)) {
    // Increment line count
    line_count++;

    // Check for comment or blank line
    char *bufferp = buffer;
    while (isspace(*bufferp)) bufferp++;
    if (*bufferp == '\0') continue;
    if (*bufferp == '#') continue;

    // Parse residue serial
    bufferp = strtok(bufferp, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse line %d of %s\n", line_count, filename); return 0; }
    int residue_serial = atoi(bufferp);

    // Parse residue letter
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse residue letter at line %d of %s\n", line_count, filename); return 0; }
    char residue_letter = *bufferp;

    // Parse residue conservation 
    bufferp = strtok(NULL, " \t");
    if (!bufferp) { fprintf(stderr, "Unable to parse mean at line %d of %s\n", line_count, filename); return 0; }
    RNScalar residue_conservation = atof(bufferp);

    // Parse residue conservation number of homologues
    // bufferp = strtok(NULL, "/");
    // if (!bufferp) { fprintf(stderr, "Unable to parse number of homologues at line %d of %s\n", line_count, filename); return 0; }
    // int residue_conservation_num_hom = atoi(bufferp);

    // Parse residue conservation total number of homologues
    // bufferp = strtok(NULL, " \t\n");
    // if (!bufferp) { fprintf(stderr, "Unable to parse total homologues at line %d of %s\n", line_count, filename); return 0; }
    // int residue_conservation_total_hom = atoi(bufferp);

    // Get residue from PDB file
    PDBResidue *residue = NULL;
    PDBAminoAcid *aminoacid = NULL;
    while (!aminoacid && (residue_count < chain->NResidues())) {
      residue = chain->Residue(residue_count);
      aminoacid = residue->AminoAcid();
      residue_count++;
    } 

    // Check residue and aminoacid
    if (!residue || !aminoacid) {
      fprintf(stderr, "Residue at line %d of %s not found in PDB file\n", line_count, filename); 
      break;
    }

    // Check if amino acid letters match 
    if (aminoacid && (aminoacid->Letter() != residue_letter)) {
      if (residue_letter == '-') {
        // Silently skip '-' in JSD file
        residue_count--;
        continue;
      }
      else if ((residue_count < chain->NResidues()) && 
               (chain->Residue(residue_count)->AminoAcid()) && 
               (chain->Residue(residue_count)->AminoAcid()->Letter() == residue_letter)) {
        // Print warning and skip residue in PDB file
        fprintf(stderr, "Skipping pdb %s %s %d (%c is not %c) in %s\n", 
          chain->Name(), residue->Name(), residue->Sequence(), aminoacid->Letter(), residue_letter, filename);
        residue = chain->Residue(residue_count);
        aminoacid = residue->AminoAcid();
        residue_count++;
      }
      else {
        // Print warning and skip residue in JSD file
        fprintf(stderr, "Skipping jsd %s %d (%c is not %c) in %s\n", 
          chain->Name(), residue_serial, aminoacid->Letter(), residue_letter, filename);
        residue_count--;
        continue;
      }      
    }

    // Assign residue conservation
    if (residue_conservation >= 0) {
      residue->conservation =  residue_conservation;
    }
  }

  // Close conservation file
  fclose(fp);

  // Normalize conservation scores
  int count = 0;
  RNScalar sum = 0;
  for (int i = 0; i < chain->NResidues(); i++) {
    PDBResidue *residue = chain->Residue(i);
    if (residue->conservation != PDB_UNKNOWN) {
      sum += residue->conservation;
      count++;
    }
  }
  if (count > 1) {
    RNScalar ssd = 0;
    RNScalar mean = sum / count;
    for (int i = 0; i < chain->NResidues(); i++) {
      PDBResidue *residue = chain->Residue(i);
      if (residue->conservation != PDB_UNKNOWN) {
        RNScalar delta = residue->conservation - mean;
        ssd += delta * delta;
      }
    }
    RNScalar variance = ssd / count;
    RNScalar stddev = sqrt(variance);
    if (stddev > 0) {
      for (int i = 0; i < chain->NResidues(); i++) {
        PDBResidue *residue = chain->Residue(i);
        if (residue->conservation != PDB_UNKNOWN) {
          residue->conservation -= mean;
          residue->conservation /= 3 * stddev;
          residue->conservation += 0.5;
          if (residue->conservation < 0) {
            residue->conservation = 0;
          }
        }
      }
    }
  }

  // Return number of residues considered
  return count;
}



#if 0

int PDBFile::
ReadJsdFiles(const char *jsd_basename)
{
  // Check number of models
  if (NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", jsd_basename);
    return 0;
  }

  // Read conservations for all chains
  RNBoolean found = 0;
  int residue_count = 0;
  char chain_filename[256];
  PDBModel *model = Model(0);
  for (int i = 0; i < model->NChains(); i++) {
    PDBChain *chain = model->Chain(i);
    char chain_name[4] = { 0 };
    strcpy(chain_name, chain->Name());
    if (chain_name[0] == ' ') chain_name[0] = '-';
    strcpy(chain_filename, jsd_basename);
    strcat(chain_filename, "_");
    strcat(chain_filename, chain_name);
    strcat(chain_filename, "_hssp.jsd_scores");
    int status = ReadJsdFile(chain, chain_filename);
    if (status == 0) return 0;
    else if (status > 0) {
      residue_count += status;
      found = 1;
    }
  }

  // Return number of residues considered
  return residue_count;
}

#else


static RNBoolean
PDBChainsAreIdentical(PDBChain *chain1, PDBChain *chain2)
{
  // Determine if chain1 and chain2 have identical sequences
  int count1 = 0;
  int count2 = 0;
  while ((count1 < chain1->NResidues()) || (count2 < chain2->NResidues())) {
    int letter1 = -1;
    while (count1 < chain1->NResidues()) {
      PDBResidue *residue1 = chain1->Residue(count1++);
      PDBAminoAcid *aminoacid1 = residue1->AminoAcid();
      if (aminoacid1) { 
        letter1 = aminoacid1->Letter(); 
        break; 
      }
    }
 
    int letter2 = -1;
    while (count2 < chain2->NResidues()) {
      PDBResidue *residue2 = chain2->Residue(count2++);
      PDBAminoAcid *aminoacid2 = residue2->AminoAcid();
      if (aminoacid2) { 
        letter2 = aminoacid2->Letter(); 
        break; 
      }
    }

    // Compare amino acids
    if ((letter1 != -1) || (letter2 != -1)) {
      if (letter1 != letter2) {
        return FALSE;
      }
    }
  }

  // Sequences are identical
  return TRUE;
}



int PDBFile::
ReadJsdFiles(const char *jsd_basename,
  const char *conservation_file_source,
  RNBoolean translate_renamed_chains,
  RNBoolean translate_unnamed_chains,
  RNBoolean translate_identical_chains)
{
  // Check number of models
  if (NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", jsd_basename);
    return 0;
  }

  // Read conservations for all chains
  int residue_count = 0;
  char chain_filename[256];
  PDBModel *model = Model(0);
  for (int i = 0; i < model->NChains(); i++) {
    PDBChain *chain1 = model->Chain(i);

    // Count residues
    int chain_residue_count = 0;
    for (int j = 0; j < chain1->NResidues(); j++) {
      PDBResidue *residue = chain1->Residue(j);
      PDBAminoAcid *aminoacid = residue->AminoAcid();
      if (!aminoacid) continue;
      if (residue->HasHetAtoms()) continue;
      chain_residue_count++;
    }

    // Check number of residues
    if (chain_residue_count < 1) continue;

    // Get original chain name
    char chain_name[4] = { 0 };
    strcpy(chain_name, chain1->Name());
    if (chain_name[0] == ' ') chain_name[0] = '-';

    // Get chain jsd filename
    strcpy(chain_filename, jsd_basename);
    strcat(chain_filename, "_");
    strcat(chain_filename, chain_name);
    // strcat(chain_filename, "_hssp.jsd_scores");
    strcat(chain_filename, "_");
    strcat(chain_filename, conservation_file_source);
    strcat(chain_filename, ".scores");

    // Read jsd file
    int status = ReadJsdFile(chain1, chain_filename);
    if (status == 0) return 0;
    else if (status > 0) {
      residue_count += status;
      continue;
    }

    // Check if chain was renamed
    if (translate_renamed_chains) {
      // Find original chain name
      for(int i = 0; i < headers.NEntries(); i++) {
        char *header = headers.Kth(i);
        char dummy[256], from[256], to[256];
        if (strstr(header, "REMARK 300 RENAMING")) {
          if (sscanf(header, "%s %s %s %s %s %s", dummy, dummy, dummy, from, dummy, to) == 6) {
            if (!strcmp(to, chain_name)) {
              strcpy(chain_name, from);
              break;
            }
          }
        }
      }

      // Get chain jsd filename
      strcpy(chain_filename, jsd_basename);
      strcat(chain_filename, "_");
      strcat(chain_filename, chain_name);
      //strcat(chain_filename, "_hssp.jsd_scores");
      strcat(chain_filename, "_");
      strcat(chain_filename, conservation_file_source);
      strcat(chain_filename, ".scores");

      // Read jsd file
      int status = ReadJsdFile(chain1, chain_filename);
      if (status == 0) return 0;
      else if (status > 0) {
        residue_count += status;
        continue;
      }
    }

    // Check jsd files for chain A if chain name is "-" or " " ???
    if (translate_unnamed_chains && (!strcmp(chain_name, "-") || !strcmp(chain_name, " "))) {
      // Get chain jsd filename
      strcpy(chain_filename, jsd_basename);
      strcat(chain_filename, "_");
      strcat(chain_filename, "A");
      // strcat(chain_filename, "_hssp.jsd_scores");
      strcat(chain_filename, "_");
      strcat(chain_filename, conservation_file_source);
      strcat(chain_filename, ".scores");

      // Read jsd file
      int status = ReadJsdFile(chain1, chain_filename);
      if (status == 0) return 0;
      else if (status > 0) {
        residue_count += status;
        continue;
      }
    }

    // Check jsd files of identical chains
    if (translate_identical_chains) {
      for (int j = 0; j < model->NChains(); j++) {
        PDBChain *chain2 = model->Chain(j);
        if (chain1 == chain2) continue;
        if (!PDBChainsAreIdentical(chain1, chain2)) continue;

        // Get original chain name
        char chain_name[4] = { 0 };
        strcpy(chain_name, chain2->Name());
        if (chain_name[0] == ' ') chain_name[0] = '-';
        for(int i = 0; i < headers.NEntries(); i++) {
          char *header = headers.Kth(i);
          char dummy[256], from[256], to[256];
          if (strstr(header, "REMARK 300 RENAMING")) {
            if (sscanf(header, "%s %s %s %s %s %s", dummy, dummy, dummy, from, dummy, to) == 6) {
              if (!strcmp(to, chain_name)) {
                strcpy(chain_name, from);
                break;
              }
            }
          }
        }


        // Get chain jsd filename
        strcpy(chain_filename, jsd_basename);
        strcat(chain_filename, "_");
        strcat(chain_filename, chain_name);
        // strcat(chain_filename, "_hssp.jsd_scores");
        strcat(chain_filename, "_");
        strcat(chain_filename, conservation_file_source);
        strcat(chain_filename, ".scores");

        // Read jsd file
        int status = ReadJsdFile(chain1, chain_filename);
        if (status == 0) return 0;
        else if (status > 0) {
          residue_count += status;
          break;
        }
      }
    }
  }

  // Return number of residues considered
  return residue_count;
}

#endif



static int 
WriteJsdFile(PDBChain *chain, char *filename)
{
  // Open conservation file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open JSD file %s\n", filename);
    return 0;
  }

  // Write lines to conservation file
  for (int i = 0; i < chain->NResidues(); i++) {
    PDBResidue *residue = chain->Residue(i);

    // Get letter
    PDBAminoAcid *amino_acid = residue->AminoAcid();
    char letter = (amino_acid) ? amino_acid->Letter() : '-';

    // Print residue to file
    fprintf(fp, "%-7d %c %13.5f 0/0\n", i, letter, residue->conservation);
  }

  // Close conservation file
  fclose(fp);

  // Return number of residues printed
  return chain->NResidues();
}



int PDBFile::
WriteJsdFiles(const char *jsd_basename, const char *conservation_file_source)
{
  // Write conservations for all chains
  int residue_count = 0;
  char chain_filename[256];
  PDBModel *model = Model(0);
  for (int i = 0; i < model->NChains(); i++) {
    PDBChain *chain = model->Chain(i);
    char chain_name[4] = { 0 };
    strcpy(chain_name, chain->Name());
    if (chain_name[0] == ' ') chain_name[0] = '-';
    strcpy(chain_filename, jsd_basename);
    strcat(chain_filename, "_");
    strcat(chain_filename, chain_name);
    // strcat(chain_filename, "_hssp.jsd_scores");
    strcat(chain_filename, "_");
    strcat(chain_filename, conservation_file_source);
    strcat(chain_filename, ".scores");
    int status = WriteJsdFile(chain, chain_filename);
    if (status == 0) return 0;
    else residue_count += status;
  }

  // Return number of residues considered
  return residue_count;
}



int PDBFile::
ReadASAFile(const char *filename)
{
  // Read asa file
  PDBFile asa(filename);
  if (!asa.ReadFile(filename)) {
    RNFail("Unable to read solvent accessible surface area file: %s", filename);
    return 0;
  }

  // Check number of models
  if (asa.NModels() != NModels()) {
    fprintf(stderr, "Number of models in ASA file does not match: %s\n", filename);
    return 0;
  }

  // Assign solvent accessible surface area for all atoms
  int atom_count = 0;
  int surface_atom_count = 0;
  for (int i = 0; i < NModels(); i++) {
    PDBModel *model = Model(i);
    PDBModel *asa_model = asa.Model(i);
    for (int j = 0; j < model->NAtoms(); j++) {
      PDBAtom *atom = model->Atom(j);
      if (atom->IsHetAtom()) continue; 
      if (atom_count < asa_model->NAtoms()) {
        PDBAtom *asa_atom = asa_model->Atom(atom_count);
        if (asa_atom->Serial() == atom->Serial()) {
          atom->accessible_surface_area = asa_atom->Occupancy();
          if (atom->accessible_surface_area > 0) surface_atom_count++;
          atom_count++;
        }
        else if (asa_atom->Serial() < atom->Serial()) {
          // My PDB parser skips some atoms (e.g., hydrogens), so be lenient to extra atoms in ASA file
          // atom_count++;
        }
        else if (asa_atom->Serial() > atom->Serial()) {
          // fprintf(stderr, "Serial number of atoms (%d %d) in ASA file does not match: %s\n", atom->Serial(), asa_atom->Serial(), filename);
          continue;
        }
      }
      else {
        // fprintf(stderr, "Number of atoms in %s is greater than in %s\n", Name(), filename);
        // return 0;
      }
    }
  }

  // Return number of surface atoms
  return surface_atom_count;
}



int PDBFile::
ReadGrowFile(const char *filename)
{
  // Get PDB model from file
  if (NModels() < 1) return 0;
  PDBModel *model = Model(0);
  if (!model) return 0;

  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open grow file: %s\n", filename);
    return 0;
  }

  // Loop while reading in records from the file 
  int line_count = 0;
  int bond_count = 0;
  char input_line[1024];
  while (fgets(input_line, 1024, fp)) {
    // Increment line count
    line_count++;

    // Check for comment
    if (input_line[0] == '#') continue;

    // Check for blank/short line
    int len = strlen(input_line);
    if (len < 62) continue;

    // Get bond type
    PDBBondType bond_type;
    switch (input_line[0]) {
      case 'H': bond_type = PDB_HYDROGEN_BOND; break;
      case 'N': bond_type = PDB_NONCOVALENT_BOND; break;
      case 'C': bond_type = PDB_COVALENT_BOND; break;
      default: bond_type = PDB_UNKNOWN_BOND; break;
    }

    // Get other information about bond (not used now)
    // int molecule1_type = input_line[2];
    // int molecule2_type = input_line[4];
    // RNScalar bond_length = atof(PDBToken(input_line, 57, 61, len));

    // Get information about atom1
    char chain1_name[4]; chain1_name[0] = input_line[6]; chain1_name[1] = '\0';
    int residue1_sequence = atoi(PDBToken(input_line, 10, 13, len));
    int residue1_icode=input_line[13];
    char residue1_name[4]; strncpy(residue1_name, PDBToken(input_line, 15, 17, len), 4);
    int atom1_serial = atoi(PDBToken(input_line, 19, 23, len));
    char atom1_name[8];  strncpy(atom1_name, &input_line[24], 4); atom1_name[4] = '\0';

    // Get information about atom2
    char chain2_name[4]; chain2_name[0] = input_line[54]; chain2_name[1] = '\0';
    int residue2_sequence = atoi(PDBToken(input_line, 49, 53, len));
    int residue2_icode=input_line[53];
    char residue2_name[4]; strncpy(residue2_name, PDBToken(input_line, 45, 47, len), 4);
    int atom2_serial = atoi(PDBToken(input_line, 39, 43, len));
    char atom2_name[8];  strncpy(atom2_name, &input_line[33], 4); atom2_name[4] = '\0';

    // Find atom1
    PDBChain *chain1 = model->FindChain(chain1_name);
    if (!chain1) { fprintf(stderr, "Unable to find chain1 (%s)at line %d in grow file %s\n", chain1_name, line_count, filename); return 0; }
    PDBResidue *residue1 = chain1->FindResidue(residue1_name, residue1_sequence, residue1_icode);
    if (!residue1) { fprintf(stderr, "Unable to find residue1 (%s %d %d) at line %d in grow file %s\n", residue1_name, residue1_sequence, residue1_icode, line_count, filename); return 0; }
    PDBAtom *atom1 = residue1->FindAtom(atom1_name, atom1_serial);
    if (!atom1) { fprintf(stderr, "Unable to find atom1 (%s %d) at line %d in grow file %s\n", atom1_name, atom1_serial, line_count, filename); return 0; }

    // Find atom2
    PDBChain *chain2 = model->FindChain(chain2_name);
    if (!chain2) { fprintf(stderr, "Unable to find chain2 (%s) at line %d in grow file %s\n", chain2_name, line_count, filename); return 0; }
    PDBResidue *residue2 = chain2->FindResidue(residue2_name, residue2_sequence, residue2_icode);
    if (!residue2) { fprintf(stderr, "Unable to find residue2 (%s %d %d) at line %d in grow file %s\n", residue2_name, residue2_sequence, residue2_icode, line_count, filename); return 0; }
    PDBAtom *atom2 = residue2->FindAtom(atom2_name, atom2_serial);
    if (!atom2) { fprintf(stderr, "Unable to find atom2 (%s %d) at line %d in grow file %s\n", atom2_name, atom2_serial, line_count, filename); return 0; }

    // Create bond
    PDBBond *bond = new PDBBond(atom1, atom2, bond_type);
    if (!bond) abort();

    // Increment number of bonds
    bond_count++;
  }

  // Close the file 
  fclose(fp);

  // Return number of bonds
  return bond_count;
}



void PDBFile::
CreateBonds(PDBResidue *ligand, RNLength max_distance)
{
  // Get model
  PDBModel *model = ligand->Model();
  if (!model) return;

  // Create bond for each pair of atoms within max_distance distance
  for (int i = 0; i < ligand->NAtoms(); i++) {
    PDBAtom *atom1 = ligand->Atom(i);
    for (int j = 0; j < model->NResidues(); j++) {
      PDBResidue *residue2 = model->Residue(j);
      if (residue2 == ligand) continue;
      if (R3Distance(atom1->Position(), residue2->BBox()) > max_distance) continue;
      for (int k = 0; k < residue2->NAtoms(); k++) {
        PDBAtom *atom2 = residue2->Atom(k);
        if (PDBDistance(atom1, atom2) > max_distance) continue;
        PDBBond *bond = new PDBBond(atom1, atom2, PDB_UNKNOWN_BOND);
        if (!bond) RNAbort("Unable to allocate bond");
      }
    }
  }
}



