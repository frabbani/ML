#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "vector.h"

/*
 Lagrangian Function
 * L(x,lambda) = f(x) + lambda*g(x)
   - dL/dx = df(x)/dx + lambda * dg(x)/dx
   - dL/dlambda = g(x)

 Finite Difference Method (FDM) using 2 point stencil
 * df(x)/dx = ( f(x+h) - f(x-h) ) / 2h

 Gradient operator (returns vector)
 * GRAD(f) = ( df/dx, df/dy )

 In this example, the input is a 2 dimensional vector
*/

#define H 0.01
#define LEARNING_RATE 0.03

// function
double f(vector_t x) {
  return x.x * x.x + x.y * x.y;
}

// constraint
double g(vector_t x) {
  return x.x + x.y + 1.0;
}

// function gradient
vector_t f_grad(vector_t v) {
  double grad_x = 0.5 / H * (f(vector(v.x + H, v.y)) - f(vector(v.x - H, v.y)));
  double grad_y = 0.5 / H * (f(vector(v.x, v.y + H)) - f(vector(v.x, v.y - H)));
  return vector(grad_x, grad_y);
}

// constraint gradient
vector_t g_grad(vector_t v) {
  double grad_x = 0.5 / H * (g(vector(v.x + H, v.y)) - g(vector(v.x - H, v.y)));
  double grad_y = 0.5 / H * (g(vector(v.x, v.y + H)) - g(vector(v.x, v.y - H)));
  return vector(grad_x, grad_y);
}

// lagrangian gradient
vector_t L_grad(vector_t v, double lambda) {
  // L(x,lambda) = f(x) + lambda*g(x)
  // GRAD(L) = GRAD(f(x)) + lambda * GRAD(g(x))
  return vector_sub(f_grad(v), vector_mul(lambda, g_grad(v)));
}

double dL_dlambda( vector_t x){
  // L(x,lambda) = f(x) + lambda*g(x)
  // dL(x)/dlambda = g(x)
  return g(x);
}


void solve_lagrange(vector_t *x_out, double *lambda_out) {
  // simple gradient decent
  vector_t x = vector(1.0, 1.0);
  double lambda = 0.0;

  vector_t grad_L, grad_f;
  int n;
  for (n = 0; n < 3000; n++) {
    grad_L = L_grad(x, lambda);
    // x = x - LR * GRAD(L)
    x = vector_sub(x, vector_mul( LEARNING_RATE, grad_L));  //descend downhill
    lambda -= LEARNING_RATE * dL_dlambda(x);

    grad_f = f_grad(x);
    if (fabs(g(x)) < TOL && fabs(grad_f.x) < TOL && fabs(grad_f.y) < TOL) {
      break;
    }
  }
  printf(" + %d steps taken!\n", n);
  *x_out = x;
  *lambda_out = lambda;
}

int main() {
  setbuf( stdout, 0);
  printf("Hello world!\n");
  printf("function f(x)...: x^2 + y^2\n");
  printf("constraint g(x).: x + y + 1 = 0\n");
  vector_t x;
  double lambda;
  solve_lagrange(&x, &lambda);
  printf("f(x) & g(x).....: %.3lf & %.3lf\n", f(x), g(x));
  printf("optimal x.......: (%.3lf, %.3lf)\n", x.x, x.y);
  printf("optimal lambda..: %.3lf\n", lambda);

  printf("Goodbye!\n");
  return 0;
}
