/* Source file for the map utilities */



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#include "R3Surfels/R3Surfels.h"
#include "R3SurfelViewer.h"
#include "map.h"


struct Map {
  RNArray<R2Point *> nodes;
};



#if 0
static R2Point
UTMPosition(double lat, double lon)
{
  // Converts a latitude/longitude pair to x and y coordinates in UTM.
  // Reference: Hoffmann-Wellenhof, B., Lichtenegger, H., and Collins, J.,
  // GPS: Theory and Practice, 3rd ed.  New York: Springer-Verlag Wien, 1994.

  // Compute lat and lon to radians
  double phi = RN_DEG2RAD(lat);
  double lambda = RN_DEG2RAD(lon);

  // Compute the UTM zone, range [1,60].
  int zone = floor ((lon + 180.0) / 6) + 1;
  
  // Compute longitude of the central meridian for the zone, in radians.
  double lambda0 = RN_DEG2RAD(-183.0 + (zone * 6.0));

  /* Ellipsoid model constants (actual values here are for WGS84) */
  double sm_a = 6378137.0;
  double sm_b = 6356752.314;

  /* Determine arc length of meridian */
  double n = (sm_a - sm_b) / (sm_a + sm_b);
  double alpha = ((sm_a + sm_b) / 2.0) * (1.0 + (pow(n, 2.0) / 4.0) + (pow(n, 4.0) / 64.0));
  double beta = (-3.0 * n / 2.0) + (9.0 * pow (n, 3.0) / 16.0) + (-3.0 * pow (n, 5.0) / 32.0);
  double gamma = (15.0 * pow (n, 2.0) / 16.0) + (-15.0 * pow (n, 4.0) / 32.0);
  double delta = (-35.0 * pow (n, 3.0) / 48.0) + (105.0 * pow (n, 5.0) / 256.0);
  double epsilon = (315.0 * pow (n, 4.0) / 512.0);
  double arc_length_of_meridian = alpha * (phi + (beta * sin (2.0 * phi))
       + (gamma * sin (4.0 * phi))
       + (delta * sin (6.0 * phi))
       + (epsilon * sin (8.0 * phi)));

  /* Calculate useful values */
  double ep2 = (pow (sm_a, 2.0) - pow (sm_b, 2.0)) / pow (sm_b, 2.0);
  double nu2 = ep2 * pow (cos (phi), 2.0);
  double N = pow (sm_a, 2.0) / (sm_b * sqrt (1 + nu2));
  double t = tan (phi);
  double t2 = t * t;
  double l = lambda - lambda0;
  double coef13 = 1.0 - t2 + nu2;
  double coef14 = 5.0 - t2 + 9 * nu2 + 4.0 * (nu2 * nu2);
  double coef15 = 5.0 - 18.0 * t2 + (t2 * t2) + 14.0 * nu2 - 58.0 * t2 * nu2;
  double coef16 = 61.0 - 58.0 * t2 + (t2 * t2) + 270.0 * nu2 - 330.0 * t2 * nu2;
  double coef17 = 61.0 - 479.0 * t2 + 179.0 * (t2 * t2) - (t2 * t2 * t2);
  double coef18 = 1385.0 - 3111.0 * t2 + 543.0 * (t2 * t2) - (t2 * t2 * t2);

  /* Calculate easting (x) */
  double x = N * cos (phi) * l
    + (N / 6.0 * pow (cos (phi), 3.0) * coef13 * pow (l, 3.0))
    + (N / 120.0 * pow (cos (phi), 5.0) * coef15 * pow (l, 5.0))
    + (N / 5040.0 * pow (cos (phi), 7.0) * coef17 * pow (l, 7.0));

  /* Calculate northing (y) */
  double y = arc_length_of_meridian
    + (t / 2.0 * N * pow (cos (phi), 2.0) * pow (l, 2.0))
    + (t / 24.0 * N * pow (cos (phi), 4.0) * coef14 * pow (l, 4.0))
    + (t / 720.0 * N * pow (cos (phi), 6.0) * coef16 * pow (l, 6.0))
    + (t / 40320.0 * N * pow (cos (phi), 8.0) * coef18 * pow (l, 8.0));

  /* Adjust easting and northing for UTM system. */
  double UTMScaleFactor = 0.9996;
  x = x * UTMScaleFactor + 500000.0;
  y = y * UTMScaleFactor;
  if (y < 0.0) y = y + 10000000.0;

  // Return point in UTM coordinates
  return R2Point(x,y);
}
#endif



////////////////////////////////////////////////////////////////////////
// Map I/O functions
////////////////////////////////////////////////////////////////////////

static char *
ParseXMLToken(const char *buffer, const char *key, int line_number, const char *filename)
{
  // Get next token
  static char keystr[1024] = { 0 };
  sprintf(keystr, "%s=\"", key);
  const char *begin = strstr(buffer, keystr);
  if (!begin) return NULL;
  static char str[32768] = { 0 };
  strncpy(str, &begin[strlen(key)+2], 1023);
  char *end = strstr(str, "\"");
  if (end) {
    *end = '\0';
    return str;
  }
  else {
    fprintf(stderr, "Error parsing %s on line %d of %s\n", key, line_number, filename); 
    fflush(stderr);
    abort();
    return NULL;
  }
}



Map *
ReadMap(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open match file: %s\n", filename);
    return 0;
  }

  // Create Map
  Map *map = new Map();
  if (!map) {
    fprintf(stderr, "Unable to allocate map for %s\n", filename);
    return NULL;
  }

  // Read lines
  int line_number = 0;
  char buffer[1024 * 1024];
  char *token;
  R2Point offset(0,0);
  while (fgets(buffer, 1024 * 1024, fp)) {
    line_number++;
    // Check if line has a node record
    if (strstr(buffer, "<node")) {
      // Parse longitude
      token = ParseXMLToken(buffer, "lon", line_number, filename);
      if (!token) continue;
      double lon = atof(token);

      // Parse lattitude
      token = ParseXMLToken(buffer, "lat", line_number, filename);
      if (!token) continue;
      double lat = atof(token);

      // Convert to UTM coordinates
      // R2Point utm = UTMPosition(lat, lon);
      // utm[0] -= 444965;
      // utm[1] -= 5029450;

      // Convert to UTM coordinates
      R2Point utm(lat, lon);

      // Insert node into map
      map->nodes.Insert(new R2Point(utm));
    }
  }

  // Close file
  fclose(fp);

  // Return map
  return map;
}



void 
DrawMap(R3SurfelViewer *viewer)
{
  // Read map
  static Map *map = NULL;
  if (!map) map = ReadMap("../../map/ottawa_manual.osm");
  if (!map) return;

  // Draw map
  glDisable(GL_LIGHTING);
  glColor3d(0, 1, 1); 
  glLineWidth(5);
  glPointSize(10);
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i < map->nodes.NEntries(); i++) {
    R2Point *position = map->nodes.Kth(i);
    R3Point center(position->X(), position->Y(), 70);
    R3LoadPoint(center);
  }
  glEnd();
  glLineWidth(1);
  glPointSize(1);
}
