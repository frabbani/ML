#pragma once

#include <math.h>

#define TOL  1e-5
#define TOL_SQ  1e-10

typedef struct {
  union{
    struct{
      double x, y;
    };
    double xy[2];
  };
} vector_t;

vector_t vector(double x, double y) {
  vector_t v;
  v.x = x;
  v.y = y;
  return v;
}

double vector_dot(vector_t u, vector_t v) {
  return u.x * v.x + u.y * v.y;
}

vector_t vector_sub(vector_t u, vector_t v) {
  u.x -= v.x;
  u.y -= v.y;
  return u;
}

vector_t vector_add(vector_t u, vector_t v) {
  u.x += v.x;
  u.y += v.y;
  return u;
}

vector_t vector_mul(double s, vector_t v) {
  v.x *= s;
  v.y *= s;
  return v;
}

vector_t vector_inv(vector_t v) {
  v.x *= -1.0;
  v.y *= -1.0;
  return v;
}

vector_t vector_madd(double s, vector_t u, vector_t v) {
  // u * s + v
  v.x += u.x * s;
  v.y += u.y * s;
  return v;
}


vector_t vector_normalized(vector_t v) {
  double s = v.x * v.x + v.y * v.y;
  if (fabs(s) > TOL_SQ) {
    s = 1.0 / sqrt(s);
    v.x *= s;
    v.y *= s;
    return v;
  }
  return vector(0.0, 0.0);
}

double vector_size_sq(vector_t v) {
  return v.x * v.x + v.y * v.y;
}

double vector_size(vector_t v) {
  double s = v.x * v.x + v.y * v.y;
  if( fabs(s) > TOL_SQ)
    return sqrt(s);
  return 0.0;
}
