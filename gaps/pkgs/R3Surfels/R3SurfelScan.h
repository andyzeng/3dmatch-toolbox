/* Include file for the R3 surfel scan class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelScan {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelScan(const char *name = NULL);
  R3SurfelScan(const R3SurfelScan& scan);

  // Destructor function
  virtual ~R3SurfelScan(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Image property functions
  int Width(void) const;
  int Height(void) const;
  int ImageWidth(void) const;
  int ImageHeight(void) const;
  const R2Point& ImageCenter(void) const;
  RNFlags Flags(void) const;
  
  // Camera property functions
  const R3CoordSystem& Pose(void) const;
  const R3Point& Viewpoint(void) const;
  R3Vector Towards(void) const;
  const R3Vector& Up(void) const;
  const R3Vector& Right(void) const;
  RNAngle FocalLength(void) const;
  RNAngle XFOV(void) const;
  RNAngle YFOV(void) const;

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;

  // Timestamp property functions
  RNScalar Timestamp(void) const;

  // Name property functions
  const char *Name(void) const;

  // User data property functions
  void *Data(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Scene access functions
  R3SurfelScene *Scene(void) const;
  int SceneIndex(void) const;

  // Node access functions
  R3SurfelNode *Node(void) const;

  // Point access functions
  R3SurfelPointSet *PointSet(RNBoolean leaf_level = FALSE) const;


  ////////////////////////////////////
  //// IMAGE GENERATION FUNCTIONS ////
  ////////////////////////////////////

  // Projection from world into image
  R2Point ImagePosition(const R3Point& world_position) const;

  // Projection of all points into image
  int RenderImage(R2Image *color_image = NULL, R2Grid *depth_image = NULL,
    R2Grid *xnormal_image = NULL, R2Grid *ynormal_image = NULL, R2Grid *znormal_image = NULL,
    R2Grid *label_image = NULL, R2Grid *object_image = NULL,
    R2Grid *node_image = NULL, R2Grid *block_image = NULL) const;

  
  /////////////////////////////////////////
  //// PROPERTY MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Pose manipulation functions
  virtual void SetPose(const R3CoordSystem& pose);
  virtual void SetViewpoint(const R3Point& viewpoint);
  virtual void SetOrientation(const R3Vector& towards, const R3Vector& up);
  virtual void SetFocalLength(RNLength focal_length);

  // Timestamp manipulation functions
  virtual void SetTimestamp(RNScalar timestamp);

  // Name manipulation functions
  virtual void SetName(const char *name);

  // Image metadata manipulation functions
  virtual void SetImageDimensions(int width, int height);
  virtual void SetImageCenter(const R2Point& center);

  // User data manipulation functions
  virtual void SetFlags(RNFlags flags);
  virtual void SetData(void *data);


  //////////////////////////////////////////
  //// STRUCTURE MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Node manipulation functions
  virtual void SetNode(R3SurfelNode *node);


  /////////////////////////////////////
  //// MEMORY MANAGEMENT FUNCTIONS ////
  /////////////////////////////////////

  // Block memory management
  void ReadBlocks(void);
  void ReleaseBlocks(void);
  RNBoolean AreBlocksResident(void) const;


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Draw function
  virtual void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

protected:
  // Internal data
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  R3SurfelNode *node;
  R3CoordSystem pose;
  RNLength focal_length;
  RNScalar timestamp;
  int image_width, image_height;
  R2Point image_center;
  char *name;
  RNFlags flags;
  void *data;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline const R3Box& R3SurfelScan::
BBox(void) const
{
  // Return bounding box of scan
  if (node) return node->BBox();
  else return R3null_box;
}



inline R3Point R3SurfelScan::
Centroid(void) const
{
  // Return centroid of scan
  if (node) return node->BBox().Centroid();
  else return R3zero_point;
}



inline const R3CoordSystem& R3SurfelScan::
Pose(void) const
{
  // Return pose 
  return pose;
}



inline const R3Point& R3SurfelScan::
Viewpoint(void) const
{
  // Return pose viewpoint
  return pose.Origin();
}



inline R3Vector R3SurfelScan::
Towards(void) const
{
  // Return pose towards vector
  return -(pose.Axes().Axis(RN_Z));
}



inline const R3Vector& R3SurfelScan::
Up(void) const
{
  // Return pose up vector
  return pose.Axes().Axis(RN_Y);
}



inline const R3Vector& R3SurfelScan::
Right(void) const
{
  // Return pose right vector
  return pose.Axes().Axis(RN_X);
}



inline RNScalar R3SurfelScan::
Timestamp(void) const
{
  // Return timestamp
  return timestamp;
}



inline RNLength R3SurfelScan::
FocalLength(void) const
{
  // Return focal length in pixels
  return focal_length;
}



inline RNLength R3SurfelScan::
XFOV(void) const
{
  // Return half-angle for horizontal field of view
  if (focal_length <= 0) return 0.0;
  return atan(0.5*image_width/focal_length);
}



inline RNLength R3SurfelScan::
YFOV(void) const
{
  // Return half-angle for vertical field of view
  if (focal_length <= 0) return 0.0;
  return atan(0.5*image_height/focal_length);
}



inline int R3SurfelScan::
ImageWidth(void) const
{
  // Return image width
  return image_width;
}



inline int R3SurfelScan::
ImageHeight(void) const
{
  // Return image height
  return image_height;
}



inline const R2Point& R3SurfelScan::
ImageCenter(void) const
{
  // Return image center
  return image_center;
}



inline const char *R3SurfelScan::
Name(void) const
{
  // Return name
  return name;
}



inline RNFlags R3SurfelScan::
Flags(void) const
{
  // Return flags
  return flags;
}



inline void *R3SurfelScan::
Data(void) const
{
  // Return user data
  return data;
}



inline R3SurfelScene *R3SurfelScan::
Scene(void) const
{
  // Return scene this scan is in
  return scene;
}



inline int R3SurfelScan::
SceneIndex(void) const
{
  // Return index in list of scans associated with scene
  return scene_index;
}



inline R3SurfelNode *R3SurfelScan::
Node(void) const
{
  // Return node
  return node;
}



inline void R3SurfelScan::
ReadBlocks(void)
{
  // Read blocks in scan
  R3SurfelNode *node = Node();
  if (!node) return;
  node->ReadBlocks();
}



inline void R3SurfelScan::
ReleaseBlocks(void)
{
  // Release blocks in scan
  R3SurfelNode *node = Node();
  if (!node) return;
  node->ReleaseBlocks();
}



inline RNBoolean R3SurfelScan::
AreBlocksResident(void) const
{
  // Return whether blocks are in memory
  R3SurfelNode *node = Node();
  if (!node) return FALSE;
  return node->AreBlocksResident();
}



