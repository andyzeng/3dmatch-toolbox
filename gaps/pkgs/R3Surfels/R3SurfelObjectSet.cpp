/* Source file for the R3 surfel object set class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelObjectSet::
R3SurfelObjectSet(void)
  : objects(),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX)
{
}



R3SurfelObjectSet::
R3SurfelObjectSet(const R3SurfelObjectSet& set)
  : objects(set.objects),
    bbox(set.bbox)
{
}



R3SurfelObjectSet::
~R3SurfelObjectSet(void)
{
}



////////////////////////////////////////////////////////////////////////
// PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////////////////

const R3Box& R3SurfelObjectSet::
BBox(void) const
{
  // Update bounding box
  if (bbox[0][0] == FLT_MAX) {
    R3SurfelObjectSet *objectset = (R3SurfelObjectSet *) this;
    objectset->bbox = R3null_box;
    for (int i = 0; i < NObjects(); i++) {
      R3SurfelObject *object = Object(i);
      objectset->bbox.Union(object->BBox());
    }
  }

  // Return bounding box
  return bbox;
}



////////////////////////////////////////////////////////////////////////
// SET MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObjectSet::
InsertObject(R3SurfelObject *object)
{
  // Insert object
  objects.Insert(object);

  // Update bounding box
  bbox[0][0] = FLT_MAX;
}



void R3SurfelObjectSet::
RemoveObject(R3SurfelObject *object)
{
  // Remove object
  RNArrayEntry *entry = objects.FindEntry(object);
  if (entry) RemoveObject(objects.EntryIndex(entry));
}



void R3SurfelObjectSet::
RemoveObject(int k)
{
  // Copy last object over object
  RNArrayEntry *entry = objects.KthEntry(k);
  assert(entry);
  R3SurfelObject *tail = objects.Tail();
  objects.EntryContents(entry) = tail;
  objects.RemoveTail();

  // Update bounding box
  bbox[0][0] = FLT_MAX;
}



void R3SurfelObjectSet::
Empty(void)
{
  // Remove objects
  objects.Empty();

  // Update bounding box
  bbox = R3Box(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObjectSet::
Draw(RNFlags flags) const
{
  // Draw objects
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    object->Draw(flags);
  }
}



void R3SurfelObjectSet::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print objects
  // Print all surfel objects
  if (prefix) fprintf(fp, "%s", prefix);
  for (int i = 0; i < NObjects(); i++) {
    R3SurfelObject *object = Object(i);
    fprintf(fp, "%d ", object->SceneIndex());
  }
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}


