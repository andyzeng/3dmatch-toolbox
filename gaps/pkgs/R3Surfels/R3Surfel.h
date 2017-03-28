/* Include file for the R3 surfel class */



/* Class definition */

class R3Surfel {
public:
  // Constructor functions
  R3Surfel(void);
  R3Surfel(float px, float py, float pz, 
    unsigned char r = 0, unsigned char g = 0, unsigned char b = 0, 
    RNBoolean aerial = FALSE);
  R3Surfel(float px, float py, float pz, 
    float nx, float ny, float nz, 
    float radius = 0, 
    unsigned char r = 0, unsigned char g = 0, unsigned char b = 0, 
    unsigned char flags = 0);

  // Position property functions
  // NOTE THAT THESE COORDINATES ARE RELATIVE TO THE BLOCK ORIGIN
  // TO GET THE WORLD COORDINATES, YOU MUST ADD THE BLOCK ORIGIN
  float X(void) const;
  float Y(void) const;
  float Z(void) const;
  float Coord(int dimension) const;
  const float *Coords(void) const;

  // Normal property functions
  float NX(void) const;
  float NY(void) const;
  float NZ(void) const;
  float NormalCoord(int dimension) const;

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

  // Manipulation functions
  void SetCoords(float x, float y, float z);
  void SetCoords(const float *xyz);
  void SetNormal(float x, float y, float z);
  void SetNormal(const float *xyz);
  void SetRadius(float radius);
  void SetColor(unsigned char r, unsigned char g, unsigned char b);
  void SetColor(const unsigned char *rgb);
  void SetColor(const RNRgb& rgb);
  void SetAerial(RNBoolean aerial = TRUE);
  void SetSilhouetteBoundary(RNBoolean boundary = TRUE);
  void SetShadowBoundary(RNBoolean boundary = TRUE);
  void SetBorderBoundary(RNBoolean boundary = TRUE);
  void SetMark(RNBoolean mark = TRUE);
  void SetActive(RNBoolean active = TRUE);
  void SetFlags(unsigned char flags);

  // Draw functions
  void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;


////////////////////////////////////////////////////////////////////////
// INTERNAL STUFF BELOW HERE
////////////////////////////////////////////////////////////////////////

  // Do not use these
  float *PositionPtr(void);
  RNInt16 *NormalPtr(void);
  RNUInt16 *RadiusPtr(void);

private:
  // Internal data
  RNScalar32 position[3];
  RNInt16 normal[3]; // x 2^15-1 (32767)
  RNUInt16 radius; // x 2^13 (8192)
  RNUChar8 color[3];
  RNUChar8 flags;
};



////////////////////////////////////////////////////////////////////////
// SURFEL FLAGS
////////////////////////////////////////////////////////////////////////

#define R3_SURFEL_ACTIVE_FLAG               0x04
#define R3_SURFEL_AERIAL_FLAG               0x02
#define R3_SURFEL_MARKED_FLAG               0x80
#define R3_SURFEL_NORMAL_FLAG               0x40
#define R3_SURFEL_SHADOW_BOUNDARY_FLAG      0x20
#define R3_SURFEL_SILHOUETTE_BOUNDARY_FLAG  0x10
#define R3_SURFEL_BORDER_BOUNDARY_FLAG      0x30
#define R3_SURFEL_BOUNDARY_FLAGS            0x30



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline float R3Surfel::
X(void) const
{
  // Return X coordinate
  return position[0];
}



inline float R3Surfel::
Y(void) const
{
  // Return Y coordinate
  return position[1];
}



inline float R3Surfel::
Z(void) const
{
  // Return Z coordinate
  return position[2];
}



inline float R3Surfel::
Coord(int dimension) const
{
  // Return coordinate
  return position[dimension];
}



inline const float *R3Surfel::
Coords(void) const
{
  // Return pointer to position
  return position;
}



inline float R3Surfel::
NX(void) const
{
  // Return X normal coordinate
  return normal[0] / 32767.0;
}



inline float R3Surfel::
NY(void) const
{
  // Return Y normal coordinate
  return normal[1] / 32767.0;
}



inline float R3Surfel::
NZ(void) const
{
  // Return Z normal coordinate
  return normal[2] / 32767.0;
}



inline float R3Surfel::
NormalCoord(int dimension) const
{
  // Return normal coordinate
  return normal[dimension] / 32767.0;
}



inline float R3Surfel::
Radius(void) const
{
  // Return radius
  return radius / 8192.0;
}



inline unsigned char R3Surfel::
R(void) const
{
  // Return red component of color
  return color[0];
}



inline unsigned char R3Surfel::
G(void) const
{
  // Return green component of color
  return color[1];
}



inline unsigned char R3Surfel::
B(void) const
{
  // Return blue component of color
  return color[2];
}



inline const unsigned char *R3Surfel::
Color(void) const
{
  // Return pointer to color
  return color;
}



inline RNRgb R3Surfel::
Rgb(void) const
{
  // Return RGB
  return RNRgb(color[0]/255.0, color[1]/255.0, color[2]/255.0);
}



inline RNBoolean R3Surfel::
IsActive(void) const
{
  // Return whether point is active (not previously deleted or subsumed by some other point)
  return flags & R3_SURFEL_ACTIVE_FLAG;
}



inline RNBoolean R3Surfel::
IsAerial(void) const
{
  // Return whether point was captured with aerial scanner
  return flags & R3_SURFEL_AERIAL_FLAG;
}



inline RNBoolean R3Surfel::
IsTerrestrial(void) const
{
  // Return whether point was captured with terrestrial scanner
  return (!IsAerial());
}



inline RNBoolean R3Surfel::
IsOnSilhouetteBoundary(void) const
{
  // Return whether point is on a silhouette boundary
  return ((flags & R3_SURFEL_BOUNDARY_FLAGS) == R3_SURFEL_SILHOUETTE_BOUNDARY_FLAG);
}



inline RNBoolean R3Surfel::
IsOnShadowBoundary(void) const
{
  // Return whether point is on a shadow boundary
  return ((flags & R3_SURFEL_BOUNDARY_FLAGS) == R3_SURFEL_SHADOW_BOUNDARY_FLAG);
}



inline RNBoolean R3Surfel::
IsOnBorderBoundary(void) const
{
  // Return whether point is on a shadow boundary
  return ((flags & R3_SURFEL_BOUNDARY_FLAGS) == R3_SURFEL_BORDER_BOUNDARY_FLAG);
}



inline RNBoolean R3Surfel::
IsOnBoundary(void) const
{
  // Return whether point is on a boundary
  return (flags & R3_SURFEL_BOUNDARY_FLAGS);
}



inline RNBoolean R3Surfel::
IsMarked(void) const
{
  // Return whether surfel is marked (useful for set and traversal operations)
  return flags & R3_SURFEL_MARKED_FLAG;
}



inline RNBoolean R3Surfel::
HasNormal(void) const
{
  // Return whether surfel has normal
  return flags & R3_SURFEL_NORMAL_FLAG;
}



inline unsigned char R3Surfel::
Flags(void) const
{
  // Return bit-encoded status flags
  return flags;
}



inline void R3Surfel::
SetCoords(float x, float y, float z)
{
  // Set position
  position[0] = x;
  position[1] = y;
  position[2] = z;
}



inline void R3Surfel::
SetCoords(const float *xyz)
{
  // Set position
  position[0] = xyz[0];
  position[1] = xyz[1];
  position[2] = xyz[2];
}



inline void R3Surfel::
SetNormal(float x, float y, float z)
{
  // Set normal
  normal[0] = (RNInt16) (32767.0 * x + 0.5);
  normal[1] = (RNInt16) (32767.0 * y + 0.5);
  normal[2] = (RNInt16) (32767.0 * z + 0.5);

  // Update flags
  flags |= R3_SURFEL_NORMAL_FLAG;
}



inline void R3Surfel::
SetNormal(const float *xyz)
{
  // Set normal
  normal[0] = (RNInt16) (32767.0 * xyz[0] + 0.5);
  normal[1] = (RNInt16) (32767.0 * xyz[1] + 0.5);
  normal[2] = (RNInt16) (32767.0 * xyz[2] + 0.5);

  // Update flags
  flags |= R3_SURFEL_NORMAL_FLAG;
}



inline void R3Surfel::
SetRadius(float radius)
{
  // Set radius
  this->radius = (RNUInt16) (8192.0 * radius + 0.5);
}



inline void R3Surfel::
SetColor(unsigned char r, unsigned char g, unsigned char b)
{
  // Set color
  color[0] = r;
  color[1] = g;
  color[2] = b;
}



inline void R3Surfel::
SetColor(const unsigned char *rgb)
{
  // Set color
  color[0] = rgb[0];
  color[1] = rgb[1];
  color[2] = rgb[2];
}



inline void R3Surfel::
SetColor(const RNRgb& rgb)
{
  // Set color
  color[0] = (unsigned char) (255.0 * rgb.R());
  color[1] = (unsigned char) (255.0 * rgb.G());
  color[2] = (unsigned char) (255.0 * rgb.B());
}



inline void R3Surfel::
SetActive(RNBoolean active)
{
  // Set whether point is active
  if (active) flags |= R3_SURFEL_ACTIVE_FLAG;
  else flags &= ~R3_SURFEL_ACTIVE_FLAG;
}



inline void R3Surfel::
SetAerial(RNBoolean aerial)
{
  // Set whether point was captured with aerial scanner
  if (aerial) flags |= R3_SURFEL_AERIAL_FLAG;
  else flags &= ~R3_SURFEL_AERIAL_FLAG;
}



inline void R3Surfel::
SetSilhouetteBoundary(RNBoolean boundary)
{
  // Set whether point is on a silhouette boundary
  if (boundary) flags |= R3_SURFEL_SILHOUETTE_BOUNDARY_FLAG;
  else flags &= ~R3_SURFEL_SILHOUETTE_BOUNDARY_FLAG;
}



inline void R3Surfel::
SetShadowBoundary(RNBoolean boundary)
{
  // Set whether point is on a shadow boundary
  if (boundary) flags |= R3_SURFEL_SHADOW_BOUNDARY_FLAG;
  else flags &= ~R3_SURFEL_SHADOW_BOUNDARY_FLAG;
}



inline void R3Surfel::
SetBorderBoundary(RNBoolean boundary)
{
  // Set whether point is on a shadow boundary
  if (boundary) flags |= R3_SURFEL_BORDER_BOUNDARY_FLAG;
  else flags &= ~R3_SURFEL_BORDER_BOUNDARY_FLAG;
}



inline void R3Surfel::
SetMark(RNBoolean mark)
{
  // Set whether surfel is marked (useful for set and traversal operations)
  if (mark) flags |= R3_SURFEL_MARKED_FLAG;
  else flags &= ~R3_SURFEL_MARKED_FLAG;
}



inline void R3Surfel::
SetFlags(unsigned char flags)
{
  // Set flags -- only use this if you REALLY know what you are doing
  this->flags = flags;
}



inline void R3Surfel::
Draw(RNFlags flags) const
{
  // Draw point at surfel
  glBegin(GL_POINTS);
  if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) glColor3ubv(color);
  glVertex3fv(position);
  glEnd();
}



////////////////////////////////////////////////////////////////////////
// Internal -- do not use
////////////////////////////////////////////////////////////////////////

inline float *R3Surfel::
PositionPtr(void)
{
  // Return pointer to position
  return position;
}



inline RNInt16 *R3Surfel::
NormalPtr(void)
{
  // Return pointer to normal
  return normal;
}



inline RNUInt16 *R3Surfel::
RadiusPtr(void)
{
  // Return pointer to radius
  return &radius;
}



