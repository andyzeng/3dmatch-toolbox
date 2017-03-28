/* Include file for the R3 surfel point class */



/* Class definition */

class R3SurfelPoint {
public:
  // Constructor functions
  R3SurfelPoint(void);
  R3SurfelPoint(const R3SurfelPoint& surfel);
  R3SurfelPoint(R3SurfelBlock *block, const R3Surfel *surfel);
  ~R3SurfelPoint(void);

  // Position property functions
  RNCoord X(void) const;
  RNCoord Y(void) const;
  RNCoord Z(void) const;
  RNCoord Coord(int dimension) const;
  R3Point Position(void) const;

  // Position property functions
  RNCoord NX(void) const;
  RNCoord NY(void) const;
  RNCoord NZ(void) const;
  RNCoord NormalCoord(int dimension) const;
  R3Vector Normal(void) const;

  // Color property functions
  unsigned char R(void) const;
  unsigned char G(void) const;
  unsigned char B(void) const;
  const unsigned char *Color(void) const;
  RNRgb Rgb(void) const;

  // Radius property functions
  float Radius(void) const;

  // Other property functions
  RNBoolean IsActive(void) const;
  RNBoolean IsMarked(void) const;
  RNBoolean IsAerial(void) const;
  RNBoolean IsTerrestrial(void) const;
  RNBoolean IsOnSilhouetteBoundary(void) const;
  RNBoolean IsOnShadowBoundary(void) const;
  RNBoolean IsOnBorderBoundary(void) const;
  RNBoolean IsOnBoundary(void) const;
  RNBoolean HasNormal(void) const;
  unsigned char Flags(void) const;

  // Access functions
  R3SurfelBlock *Block(void) const;
  int BlockIndex(void) const;
  const R3Surfel *Surfel(void) const;

  // Copy and reset functions
  void Copy(const R3SurfelPoint *point);
  void Reset(R3SurfelBlock *block, const R3Surfel *surfel);
  R3SurfelPoint& operator=(const R3SurfelPoint& point);

  // Manipulation functions
  void SetPosition(const R3Point& position);
  void SetNormal(const R3Vector& normal);
  void SetColor(const RNRgb& color);
  void SetActive(RNBoolean active = TRUE);
  void SetAerial(RNBoolean aerial = TRUE);
  void SetMark(RNBoolean mark);

  // Draw functions
  void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;

private:
  R3SurfelBlock *block;
  const R3Surfel *surfel;
};



/* Public function declarations */

extern R3Point SurfelPointPosition(R3SurfelPoint *point, void *);



/* Inline functions */

inline RNCoord R3SurfelPoint::
X(void) const
{
  // Return X coordinate of surfel point in global coordinate system
  assert(block && surfel);
 return surfel->X() + block->Origin().X();
}



inline RNCoord R3SurfelPoint::
Y(void) const
{
  // Return Y coordinate of surfel point in global coordinate system
  assert(block && surfel);
  return surfel->Y() + block->Origin().Y();
}



inline RNCoord R3SurfelPoint::
Z(void) const
{
  // Return Z coordinate of surfel point in global coordinate system
  assert(block && surfel);
  return surfel->Z() + block->Origin().Z();
}



inline RNCoord R3SurfelPoint::
Coord(int dimension) const
{
  // Return coordinate of surfel point in global coordinate system
  assert(block && surfel);
  return surfel->Coord(dimension) + block->Origin().Coord(dimension);
}



inline R3Point R3SurfelPoint::
Position(void) const
{
  // Return position of surfel point in global coordinate system
  return R3Point(X(), Y(), Z());
}



inline RNCoord R3SurfelPoint::
NX(void) const
{
  // Return X normal of surfel 
  assert(block && surfel);
 return surfel->NX();
}



inline RNCoord R3SurfelPoint::
NY(void) const
{
  // Return Y normal of surfel 
  assert(block && surfel);
  return surfel->NY();
}



inline RNCoord R3SurfelPoint::
NZ(void) const
{
  // Return Z normal of surfel 
  assert(block && surfel);
  return surfel->NZ();
}



inline RNCoord R3SurfelPoint::
NormalCoord(int dimension) const
{
  // Return normal coordinate of surfel point 
  assert(block && surfel);
  return surfel->NormalCoord(dimension);
}



inline R3Vector R3SurfelPoint::
Normal(void) const
{
  // Return position of surfel point in global coordinate system
  return R3Vector(NX(), NY(), NZ());
}



inline float R3SurfelPoint::
Radius(void) const
{
  // Return radius of surfel
  return surfel->Radius();
}



inline unsigned char R3SurfelPoint::
R(void) const
{
  // Return red component of color
  return surfel->R();
}



inline unsigned char R3SurfelPoint::
G(void) const
{
  // Return green component of color
  return surfel->G();
}



inline unsigned char R3SurfelPoint::
B(void) const
{
  // Return blue component of color
  return surfel->B();
}



inline const unsigned char *R3SurfelPoint::
Color(void) const
{
  // Return pointer to color
  return surfel->Color();
}



inline RNRgb R3SurfelPoint::
Rgb(void) const
{
  // Return RGB
  return surfel->Rgb();
}


inline RNBoolean R3SurfelPoint::
IsActive(void) const
{
  // Return whether point is active
  return surfel->IsActive();
}



inline RNBoolean R3SurfelPoint::
IsMarked(void) const
{
  // Return whether point is marked
  return surfel->IsMarked();
}



inline RNBoolean R3SurfelPoint::
IsAerial(void) const
{
  // Return whether point was captured with aerial scanner
  return surfel->IsAerial();
}



inline RNBoolean R3SurfelPoint::
IsTerrestrial(void) const
{
  // Return whether point was captured with terrestrial scanner
  return surfel->IsTerrestrial();
}



inline RNBoolean R3SurfelPoint::
IsOnSilhouetteBoundary(void) const
{
  // Return whether point is on silhouette boundary
  return surfel->IsOnSilhouetteBoundary();
}



inline RNBoolean R3SurfelPoint::
IsOnShadowBoundary(void) const
{
  // Return whether point is on shadow boundary
  return surfel->IsOnShadowBoundary();
}



inline RNBoolean R3SurfelPoint::
IsOnBorderBoundary(void) const
{
  // Return whether point is on border boundary
  return surfel->IsOnBorderBoundary();
}



inline RNBoolean R3SurfelPoint::
IsOnBoundary(void) const
{
  // Return whether point is on boundary
  return surfel->IsOnBoundary();
}



inline RNBoolean R3SurfelPoint::
HasNormal(void) const
{
  // Return whether point has normal
  return surfel->HasNormal();
}



inline unsigned char R3SurfelPoint::
Flags(void) const
{
  // Return bit-encoded status flags
  return surfel->Flags();
}



inline R3SurfelBlock *R3SurfelPoint::
Block(void) const
{
  // Return block
  return block;
}



inline int R3SurfelPoint::
BlockIndex(void) const
{
  // Return index of surfel within block
  return surfel - block->Surfels();
}



inline const R3Surfel *R3SurfelPoint::
Surfel(void) const
{
  // Return surfel 
  return surfel;
}



