/* Source file for the R3 surfel class */



/* Include files */

#include "R3Surfels/R3Surfels.h"



/* Public functions */

R3Surfel::
R3Surfel(void)
  : radius(0),
    flags(0)
{
  // Set everything
  this->position[0] = 0;
  this->position[1] = 0;
  this->position[2] = 0;
  this->normal[0] = 0;
  this->normal[1] = 0;
  this->normal[2] = 0;
  this->color[0] = 0;
  this->color[1] = 0;
  this->color[2] = 0;
}



R3Surfel::
R3Surfel(float x, float y, float z, 
  unsigned char r, unsigned char g, unsigned char b, 
  RNBoolean aerial)
  : radius(0),
    flags(0)
{
  // Set everything
  this->position[0] = x;
  this->position[1] = y;
  this->position[2] = z;
  this->normal[0] = 0;
  this->normal[1] = 0;
  this->normal[2] = 0;
  this->color[0] = r;
  this->color[1] = g;
  this->color[2] = b;
  SetAerial(aerial);
}



R3Surfel::
R3Surfel(float x, float y, float z, float nx, float ny, float nz,
  float radius, unsigned char r, unsigned char g, unsigned char b, 
  unsigned char flags)
  : flags(flags)
{
  // Set everything
  this->position[0] = x;
  this->position[1] = y;
  this->position[2] = z;
  this->normal[0] = (RNInt16) (32767.0 * nx + 0.5);
  this->normal[1] = (RNInt16) (32767.0 * ny + 0.5);
  this->normal[2] = (RNInt16) (32767.0 * nz + 0.5);
  this->color[0] = r;
  this->color[1] = g;
  this->color[2] = b;
  this->radius = (RNUInt16) (8192.0 * radius + 0.5);
}



