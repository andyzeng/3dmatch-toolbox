// Source file for the scene information program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

// Input scene file
static char *input_scene_name = NULL;
static char *input_categories_name = NULL;

// What to print
static int print_nodes = 0;
static int print_elements = 0;
static int print_references = 0;
static int print_lights = 0;
static int print_brdfs = 0;
static int print_textures = 0;
static int print_materials = 0;

// Scene processing 
static int remove_references = 0;
static int remove_hierarchy = 0;
static int remove_transformations = 0;



////////////////////////////////////////////////////////////////////////
// Input functions
////////////////////////////////////////////////////////////////////////

static R3Scene *
ReadScene(char *filename)
{
  // Allocate scene
  R3Scene *scene = new R3Scene();
  if (!scene) {
    fprintf(stderr, "Unable to allocate scene for %s\n", filename);
    return NULL;
  }

  // Read scene from file
  if (!scene->ReadFile(filename)) {
    delete scene;
    return NULL;
  }

  // Process scene
  if (remove_references) scene->RemoveReferences();
  if (remove_transformations) scene->RemoveTransformations();
  if (remove_hierarchy) scene->RemoveHierarchy();
  
  // Return scene
  return scene;
}



static int
ReadCategories(R3Scene *scene, const char *filename)
{
  // Read file
  if (!scene->ReadSUNCGModelFile(filename)) return 0;

  // Return success
  return 1;
} 



////////////////////////////////////////////////////////////////////////
// Printing functions
////////////////////////////////////////////////////////////////////////

static int
PrintBrdf(R3Brdf *brdf, int index, int level)
{
  // Construct indent
  char indent[4096] = { '\0' };
  for (int i = 0; i < level; i++) {
    strcat(indent, "  ");
  }

  // Print brdf
  const char *name = brdf->Name();
  const RNRgb& ambient = brdf->Ambient();
  const RNRgb& diffuse = brdf->Diffuse();
  const RNRgb& specular = brdf->Specular();
  const RNRgb& transmission = brdf->Transmission();
  const RNRgb& emission = brdf->Emission();
  printf("%sBrdf %d\n", indent, index);
  if (name) printf("%s  Name = %s\n", indent, name);
  if (!ambient.IsBlack()) printf("%s  Ambient = %g %g %g\n", indent, ambient.R(), ambient.G(), ambient.B());
  if (!diffuse.IsBlack()) printf("%s  Diffuse = %g %g %g\n", indent, diffuse.R(), diffuse.G(), diffuse.B());
  if (!specular.IsBlack()) printf("%s  Specular = %g %g %g\n", indent, specular.R(), specular.G(), specular.B());
  if (!transmission.IsBlack()) printf("%s  Transmission = %g %g %g\n", indent, transmission.R(), transmission.G(), transmission.B());
  if (!emission.IsBlack()) printf("%s  Emission = %g %g %g\n", indent, emission.R(), emission.G(), emission.B());
  if (brdf->Shininess() != 0) printf("%s  Shininess = %g\n", indent, brdf->Shininess());
  if (brdf->IndexOfRefraction() != 1) printf("%s  Index of refraction = %g\n", indent, brdf->IndexOfRefraction());
  printf("\n");

  // Return success
  return 1;
}



static int
PrintTexture(R2Texture *texture, int index, int level)
{
  // Construct indent
  char indent[4096] = { '\0' };
  for (int i = 0; i < level; i++) {
    strcat(indent, "  ");
  }

  // Print texture
  const char *name = texture->Name();
  const char *filename = texture->Filename();
  const R2Image *image = texture->Image();
  int width = (image) ? image->Width() : 0;
  int height = (image) ? image->Height() : 0;
  int ncomponents = (image) ? image->NComponents() : 0;
  printf("%sTexture %d\n", indent, index);
  if (name) printf("%s  Name = %s\n", indent, name);
  printf("%s  Scene index = %d\n", indent, texture->SceneIndex());
  printf("%s  Dimensions = %d %d\n", indent, width, height);
  printf("%s  NComponents = %d\n", indent, ncomponents);
  if (filename) printf("%s  Filename = %s\n", indent, filename);
  printf("\n");

  // Return success
  return 1;
}



static int
PrintMaterial(R3Material *material, int index, int level)
{
  // Construct indent
  char indent[4096] = { '\0' };
  for (int i = 0; i < level; i++) {
    strcat(indent, "  ");
  }

  // Print material
  const char *name = material->Name();
  const R3Brdf *brdf = material->Brdf();
  const char *brdf_name = (brdf) ? brdf->Name() : NULL;
  const R2Texture *texture = material->Texture();
  const char *texture_name = (texture) ? texture->Name() : NULL;
  const RNRgb& diffuse = (brdf) ? brdf->Diffuse() : RNblack_rgb;
  printf("%sMaterial %d\n", indent, index);
  if (name) printf("%s  Name = %s\n", indent, name);
  if (texture) printf("%s  Texture index = %d\n", indent, texture->SceneIndex());
  if (texture_name) printf("%s  Texture name = %s\n", indent, texture_name);
  if (brdf) printf("%s  Brdf index = %d\n", indent, brdf->SceneIndex());
  if (brdf_name) printf("%s  Brdf name = %s\n", indent, brdf_name);
  printf("%s  Brdf diffuse = %g %g %g\n", indent, diffuse.R(), diffuse.G(), diffuse.B());
  printf("\n");

  // Return success
  return 1;
}



static int
PrintLight(R3Light *light, int index, int level)
{
  // Construct indent
  char indent[4096] = { '\0' };
  for (int i = 0; i < level; i++) {
    strcat(indent, "  ");
  }

  // Print light
  const RNRgb& color = light->Color();
  printf("%sLight %d\n", indent, index);
  printf("%s  Intensity = %g\n", indent, light->Intensity());
  printf("%s  Color = %g %g %g\n", indent, color.R(), color.G(), color.B());
  printf("\n");

  // Return success
  return 1;
}



static int
PrintElement(R3SceneElement *element, int index, int level)
{
  // Construct indent
  char indent[4096] = { '\0' };
  for (int i = 0; i < level; i++) {
    strcat(indent, "  ");
  }

  // Print element stuff
  R3Box bbox = element->BBox();
  R3Point centroid = element->Centroid();
  R3Material *material = element->Material();
  printf("%sElement %d\n", indent, index);
  printf("%s  # Shapes = %d\n", indent, element->NShapes());
  if (material) printf("%s  Material = %d\n", indent, material->SceneIndex());
  printf("%s  BBox = ( %g %g %g ) ( %g %g %g )\n", indent, bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
  printf("%s  Centroid = %g %g %g\n", indent, centroid[0], centroid[1], centroid[2]);
  printf("\n");

  // Return success
  return 1;
}



static int
PrintReference(R3SceneReference *reference, int index, int level)
{
  // Construct indent
  char indent[4096] = { '\0' };
  for (int i = 0; i < level; i++) {
    strcat(indent, "  ");
  }

  // Print reference stuff
  const R3Scene *referenced_scene = reference->ReferencedScene();
  const char *name = (referenced_scene) ? referenced_scene->Root()->Name() : NULL;
  const char *filename = (referenced_scene) ? referenced_scene->Filename() : NULL;
  printf("%sReference %d\n", indent, index);
  if (name) printf("%s  Name = %s\n", indent, name);
  if (filename) printf("%s  Filename = %s\n", indent, filename);
  printf("%s  # Materials = %d\n", indent, reference->NMaterials());
  for (int i = 0; i < reference->NMaterials(); i++) {
    R3Material *material = reference->Material(i);
    const R3Brdf *brdf = (material) ? material->Brdf() : NULL;
    const R2Texture *texture = (material) ? material->Texture() : NULL;
    RNRgb diffuse = (brdf) ? brdf->Diffuse() : RNRgb(0.5, 0.5, 0.5);
    const char *texture_name = (texture) ? texture->Name() : "None";
    printf("%s    %d (%g %g %g) %s\n", indent, i, diffuse.R(), diffuse.G(), diffuse.B(), texture_name);
  } 
  printf("\n");

  // Return success
  return 1;
}



static int
PrintNodes(R3Scene *scene, R3SceneNode *node, int index, int level)
{
  // Construct indent
  char indent[4096] = { '\0' };
  for (int i = 0; i < level; i++) {
    strcat(indent, "  ");
  }

  // Print node
  const char *name = node->Name();
  R3Box bbox = node->BBox();
  R3Point centroid = node->Centroid();
  R4Matrix m = node->Transformation().Matrix();
  printf("%sNode %s\n", indent, (name) ? name : "Null");
  printf("%s  Parent  = %s\n", indent, (node->Parent()) ? ((node->Parent()->Name()) ? node->Parent()->Name() : "NoName") : "None");
  printf("%s  Parent index = %d\n", indent, node->ParentIndex());
  printf("%s  Scene index = %d\n", indent, node->SceneIndex());
  printf("%s  # Elements = %d\n", indent, node->NElements());
  printf("%s  # References = %d\n", indent, node->NReferences());
  printf("%s  # Facets = %g\n", indent, node->NFacets().Mid());
  printf("%s  Area = %g\n", indent, node->Area());
  printf("%s  Centroid = %g %g %g\n", indent, centroid[0], centroid[1], centroid[2]);
  printf("%s  BBox = ( %g %g %g ) ( %g %g %g )\n", indent, bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
  printf("%s  Transformation matrix = \n", indent);
  printf("%s    %12.3g %12.3g %12.3g %12.3g\n", indent, m[0][0], m[0][1], m[0][2], m[0][3]);
  printf("%s    %12.3g %12.3g %12.3g %12.3g\n", indent, m[1][0], m[1][1], m[1][2], m[1][3]);
  printf("%s    %12.3g %12.3g %12.3g %12.3g\n", indent, m[2][0], m[2][1], m[2][2], m[2][3]);
  printf("%s    %12.3g %12.3g %12.3g %12.3g\n", indent, m[3][0], m[3][1], m[3][2], m[3][3]);
  printf("\n");

  // Print elements
  if (print_elements) {
    for (int i = 0; i < node->NElements(); i++) {
      R3SceneElement *element = node->Element(i);
      if (!PrintElement(element, i, level+1)) return 0;
    }
  }

  // Print references
  if (print_references) {
    for (int i = 0; i < node->NReferences(); i++) {
      R3SceneReference *reference = node->Reference(i);
      if (!PrintReference(reference, i, level+1)) return 0;
    }
  }

  // Print children
  for (int i = 0; i < node->NChildren(); i++) {
    R3SceneNode *child = node->Child(i);
    if (!PrintNodes(scene, child, i, level+1)) return 0;
  }

  // Return success
  return 1;
}



static int
PrintScene(R3Scene *scene)
{
  // Print scene stuff
  const R3Camera& camera = scene->Camera();
  const R3Box& bbox = scene->BBox();
  R3Point centroid = scene->Centroid();
  printf("Scene ...\n");
  printf("  # Nodes = %d\n", scene->NNodes());
  printf("  # Lights = %d\n", scene->NLights());
  printf("  # Materials = %d\n", scene->NMaterials());
  printf("  # Brdfs = %d\n", scene->NBrdfs());
  printf("  # Textures = %d\n", scene->NTextures());
  printf("  # Referenced scenes = %d\n", scene->NReferencedScenes());
  printf("  Camera = %g %g %g   %g %g %g   %g %g %g  %g\n", 
    camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
    camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
    camera.Up().X(), camera.Up().Y(), camera.Up().Z(),
    camera.YFOV());
  printf("  BBox = ( %g %g %g ) ( %g %g %g )\n", 
    bbox[0][0], bbox[0][1], bbox[0][2], 
    bbox[1][0], bbox[1][1], bbox[1][2]);
  printf("  Centroid = %g %g %g\n", 
    centroid[0], centroid[1], centroid[2]);
  printf("\n");

  // Print nodes and elements and references in a hierarchy
  if (print_nodes || print_elements || print_references) {
    printf("Nodes ...\n");
    PrintNodes(scene, scene->Root(), 0, 0);
  }

  // Print lights
  if (print_lights) {
    printf("Lights ...\n");
    for (int i = 0; i < scene->NLights(); i++) {
      R3Light *light = scene->Light(i);
      PrintLight(light, i, 1);
    }
  }

  // Print brdfs
  if (print_brdfs) {
    printf("Brdfs ...\n");
    for (int i = 0; i < scene->NBrdfs(); i++) {
      R3Brdf *brdf = scene->Brdf(i);
      PrintBrdf(brdf, i, 1);
    }
  }

  // Print textures
  if (print_textures) {
    printf("Textures ...\n");
    for (int i = 0; i < scene->NTextures(); i++) {
      R2Texture *texture = scene->Texture(i);
      PrintTexture(texture, i, 1);
    }
  }

  // Print materials
  if (print_materials) {
    printf("Materials ...\n");
    for (int i = 0; i < scene->NMaterials(); i++) {
      R3Material *material = scene->Material(i);
      PrintMaterial(material, i, 1);
    }
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Program argument parsing function
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-nodes")) print_nodes = 1; 
      else if (!strcmp(*argv, "-nodes")) print_nodes = 1; 
      else if (!strcmp(*argv, "-elements")) print_elements = 1; 
      else if (!strcmp(*argv, "-references")) print_references = 1; 
      else if (!strcmp(*argv, "-lights")) print_lights = 1; 
      else if (!strcmp(*argv, "-brdfs")) print_brdfs = 1; 
      else if (!strcmp(*argv, "-textures")) print_textures = 1; 
      else if (!strcmp(*argv, "-materials")) print_materials = 1; 
      else if (!strcmp(*argv, "-remove_references")) remove_references = 1; 
      else if (!strcmp(*argv, "-remove_hierarchy")) remove_hierarchy = 1; 
      else if (!strcmp(*argv, "-remove_transformations")) remove_transformations = 1;
      else if (!strcmp(*argv, "-categories")) { argc--; argv++; input_categories_name = *argv; }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input_scene_name) input_scene_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check scene filename
  if (!input_scene_name) {
    fprintf(stderr, "Usage: scninfo inputscenefile [options]\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Main function
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read scene
  R3Scene *scene = ReadScene(input_scene_name);
  if (!scene) exit(-1);

  // Read categories
  if (input_categories_name) {
    if (!ReadCategories(scene, input_categories_name)) exit(-1);
  }

  // Print scene
  if (!PrintScene(scene)) exit(-1);

  // Return success 
  return 0;
}



