/* Include file for the R3 surfel scene class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelScene {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelScene(const char *name = NULL);
  R3SurfelScene(const R3SurfelScene& scene);

  // Destructor function
  virtual ~R3SurfelScene(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;

  // Name property functions
  const char *Name(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Tree access functions
  R3SurfelTree *Tree(void) const;

  // Object access functions
  int NObjects(void) const;
  R3SurfelObject *Object(int k) const;
  R3SurfelObject *FindObjectByName(const char *object_name) const;
  R3SurfelObject *FindObjectByIdentifier(int identifier) const;
  R3SurfelObject *RootObject(void) const;

  // Label access functions
  int NLabels(void) const;
  R3SurfelLabel *Label(int k) const;
  R3SurfelLabel *FindLabelByName(const char *label_name) const;
  R3SurfelLabel *FindLabelByIdentifier(int identifier) const;
  R3SurfelLabel *FindLabelByAssignmentKeystroke(int key) const;
  R3SurfelLabel *RootLabel(void) const;

  // Object property access functions
  int NObjectProperties(void) const;
  R3SurfelObjectProperty *ObjectProperty(int k) const;

  // Label property access functions
  int NLabelProperties(void) const;
  R3SurfelLabelProperty *LabelProperty(int k) const;

  // Object relationship access functions
  int NObjectRelationships(void) const;
  R3SurfelObjectRelationship *ObjectRelationship(int k) const;

  // Label relationship access functions
  int NLabelRelationships(void) const;
  R3SurfelLabelRelationship *LabelRelationship(int k) const;

  // Label assignment access functions
  int NLabelAssignments(void) const;
  R3SurfelLabelAssignment *LabelAssignment(int k) const;
  R3SurfelLabelAssignment *FindLabelAssignment(R3SurfelObject *object, R3SurfelLabel *label, RNScalar confidence, int originator) const;

  // Scan access functions
  int NScans(void) const;
  R3SurfelScan *Scan(int k) const;
  R3SurfelScan *FindScanByName(const char *scan_name) const;

  // Feature access functions
  int NFeatures(void) const;
  R3SurfelFeature *Feature(int k) const;
  R3SurfelFeature *FindFeatureByName(const char *feature_name) const;


  /////////////////////////////////////////
  //// PROPERTY MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Name manipulation functions
  virtual void SetName(const char *name);


  //////////////////////////////////////////
  //// STRUCTURE MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Object mainpulation functions
  virtual void InsertObject(R3SurfelObject *object, R3SurfelObject *parent);
  virtual void RemoveObject(R3SurfelObject *object);
  virtual void MergeObject(R3SurfelObject *dst_object, R3SurfelObject *src_object);

  // Label manipulation functions
  virtual void InsertLabel(R3SurfelLabel *label, R3SurfelLabel *parent);
  virtual void RemoveLabel(R3SurfelLabel *label);

  // Object property manipulation functions
  virtual void InsertObjectProperty(R3SurfelObjectProperty *property);
  virtual void RemoveObjectProperty(R3SurfelObjectProperty *property);

  // Label property manipulation functions
  virtual void InsertLabelProperty(R3SurfelLabelProperty *property);
  virtual void RemoveLabelProperty(R3SurfelLabelProperty *property);

  // Object relationship manipulation functions
  virtual void InsertObjectRelationship(R3SurfelObjectRelationship *relationship);
  virtual void RemoveObjectRelationship(R3SurfelObjectRelationship *relationship);

  // Label relationship manipulation functions
  virtual void InsertLabelRelationship(R3SurfelLabelRelationship *relationship);
  virtual void RemoveLabelRelationship(R3SurfelLabelRelationship *relationship);

  // Label assignment manipulation functions
  virtual void InsertLabelAssignment(R3SurfelLabelAssignment *assignment);
  virtual void RemoveLabelAssignment(R3SurfelLabelAssignment *assignment);

  // Scan manipulation functions
  virtual void InsertScan(R3SurfelScan *scan);
  virtual void RemoveScan(R3SurfelScan *scan);

  // Feature manipulation functions
  virtual void InsertFeature(R3SurfelFeature *feature);
  virtual void RemoveFeature(R3SurfelFeature *feature);

  // Scene merging
  virtual void InsertScene(const R3SurfelScene& scene2, 
    R3SurfelObject *parent_object1 = NULL, 
    R3SurfelLabel *parent_label1 = NULL,
    R3SurfelNode *parent_node1 = NULL);


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Draw function
  virtual void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ///////////////////////
  //// I/O FUNCTIONS ////
  ///////////////////////

  // I/O functions
  virtual int OpenFile(const char *scene_filename, const char *database_filename, 
    const char *scene_rwaccess = NULL, const char *database_rwaccess = NULL);
  virtual int SyncFile(const char *output_scene_filename = NULL);
  virtual int CloseFile(const char *output_scene_filename = NULL);
  void SetDirty(void);

  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

  // Flag constants
# define R3_SURFEL_SCENE_DIRTY_FLAG   0x01
 
public:
  // File I/O functions
  virtual int ReadFile(const char *filename);
  virtual int WriteFile(const char *filename);
  virtual int ReadAsciiFile(const char *filename);
  virtual int WriteAsciiFile(const char *filename);
  virtual int ReadBinaryFile(const char *filename);
  virtual int WriteBinaryFile(const char *filename);
  virtual int WriteARFFFile(const char *filename);
  virtual int WriteTianqiangFile(const char *filename);

protected:
  // Structure access stuff
  R3SurfelTree *tree;
  RNArray<R3SurfelObject *> objects;
  RNArray<R3SurfelLabel *> labels;
  RNArray<R3SurfelObjectProperty *> object_properties;
  RNArray<R3SurfelLabelProperty *> label_properties;
  RNArray<R3SurfelObjectRelationship *> object_relationships;
  RNArray<R3SurfelLabelRelationship *> label_relationships;
  RNArray<R3SurfelLabelAssignment *> assignments;
  RNArray<R3SurfelScan *> scans;
  RNArray<R3SurfelFeature *> features;

  // File stuff
  char *filename;
  char *rwaccess;
  char *name;
  RNFlags flags;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline const R3Box& R3SurfelScene::
BBox(void) const
{
  // Return bounding box of scene
  return tree->BBox();
}



inline R3Point R3SurfelScene::
Centroid(void) const
{
  // Return centroid of scene
  return BBox().Centroid();
}



inline const char *R3SurfelScene::
Name(void) const
{
  // Return name
  return name;
}



inline R3SurfelTree *R3SurfelScene::
Tree(void) const
{
  // Return surfel tree
  return tree;
}



inline int R3SurfelScene::
NObjects(void) const
{
  // Return number of objects
  return objects.NEntries();
}



inline R3SurfelObject *R3SurfelScene::
Object(int k) const
{
  // Return kth object
  return objects.Kth(k);
}



inline R3SurfelObject *R3SurfelScene::
RootObject(void) const
{
  // Return root object
  return objects.Kth(0);
}



inline int R3SurfelScene::
NLabels(void) const
{
  // Return number of labels
  return labels.NEntries();
}



inline R3SurfelLabel *R3SurfelScene::
Label(int k) const
{
  // Return kth label
  return labels.Kth(k);
}



inline R3SurfelLabel *R3SurfelScene::
RootLabel(void) const
{
  // Return root label
  return labels.Kth(0);
}



inline int R3SurfelScene::
NObjectProperties(void) const
{
  // Return number of object properties
  return object_properties.NEntries();
}



inline R3SurfelObjectProperty *R3SurfelScene::
ObjectProperty(int k) const
{
  // Return kth object property
  return object_properties.Kth(k);
}



inline int R3SurfelScene::
NLabelProperties(void) const
{
  // Return number of label properties
  return label_properties.NEntries();
}



inline R3SurfelLabelProperty *R3SurfelScene::
LabelProperty(int k) const
{
  // Return kth label property
  return label_properties.Kth(k);
}



inline int R3SurfelScene::
NObjectRelationships(void) const
{
  // Return number of object relationships
  return object_relationships.NEntries();
}



inline R3SurfelObjectRelationship *R3SurfelScene::
ObjectRelationship(int k) const
{
  // Return kth object relationship
  return object_relationships.Kth(k);
}



inline int R3SurfelScene::
NLabelRelationships(void) const
{
  // Return number of label relationships
  return label_relationships.NEntries();
}



inline R3SurfelLabelRelationship *R3SurfelScene::
LabelRelationship(int k) const
{
  // Return kth label relationship
  return label_relationships.Kth(k);
}



inline int R3SurfelScene::
NLabelAssignments(void) const
{
  // Return number of assignments
  return assignments.NEntries();
}



inline R3SurfelLabelAssignment *R3SurfelScene::
LabelAssignment(int k) const
{
  // Return kth assignment
  return assignments[k];
}



inline int R3SurfelScene::
NScans(void) const
{
  // Return number of scans
  return scans.NEntries();
}



inline R3SurfelScan *R3SurfelScene::
Scan(int k) const
{
  // Return kth scan
  return scans.Kth(k);
}



inline int R3SurfelScene::
NFeatures(void) const
{
  // Return number of features
  return features.NEntries();
}



inline R3SurfelFeature *R3SurfelScene::
Feature(int k) const
{
  // Return kth feature
  return features.Kth(k);
}



inline void R3SurfelScene::
SetDirty(void)
{
  // Mark scene as dirty
  flags.Add(R3_SURFEL_SCENE_DIRTY_FLAG);
}



