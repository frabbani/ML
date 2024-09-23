#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "vector.h"
#include "expr.h"
typedef struct expr_var expr_var_t;

/*
 Lagrangian Function
 * L(x,lambda) = f(x) + lambda*g(x)
 * when minimizing
 - dL/dx = df(x)/dx - lambda * dg(x)/dx
 - x = x - learning_rate * dL/dx
 * when maximizing
 - dL/dx = df(x)/dx + lambda * dg(x)/dx
 - x = x + learning_rate * dL/dx
 * lambda -= learning_rate * g(x)
 - dL/dlambda = g(x)

 Finite Difference Method (FDM) using 2 point stencil
 * df(x)/dx = ( f(x+h) - f(x-h) ) / 2h

 Gradient operator (returns vector)
 * GRAD(f) = ( df/dx, df/dy )

 In this example, the input is a 2 dimensional vector
 */

#define H 0.01
#define LEARNING_RATE 0.03

vector_t x_eval;
struct expr_var_list fofx_vars = { 0 };
struct expr_var_list gofx_vars = { 0 };
struct expr *fofx_expr = NULL;
struct expr *gofx_expr = NULL;
double constraint = 0.0;

double getx() {
  return x_eval.x;
}
double gety() {
  return x_eval.y;
}

static struct expr_func user_funcs[] = { { "getx", getx, NULL, 0 }, { "gety",
    gety, NULL, 0 }, { NULL, NULL, NULL, 0 }, };


void init() {
  char *str = malloc(64 * 1024);
  char line[256];
  FILE *fp = NULL;

  str[0] = '\0';
  fp = fopen("fofx.txt", "r");
  if (fp) {
    while (fgets(line, sizeof(line), fp))
      if ('#' != line[0])
        strcat(str, line);
    fclose(fp);
  }

  fofx_expr = expr_create(str, strlen(str), &fofx_vars, user_funcs);
  if (fofx_expr) {
    printf("f(x) loaded\n");
  }

  str[0] = '\0';
  fp = fopen("gofx.txt", "r");
  if (fp) {
    while (fgets(line, sizeof(line), fp))
      if ('#' != line[0])
        strcat(str, line);
    fclose(fp);
  }
  gofx_expr = expr_create(str, strlen(str), &gofx_vars, user_funcs);
  if (gofx_expr) {
    printf("g(x) loaded\n");
  }

  free(str);
}

void term() {
  if (fofx_expr)
    expr_destroy(fofx_expr, &fofx_vars);
  if (gofx_expr)
    expr_destroy(gofx_expr, &gofx_vars);
}

// function
double f(vector_t x) {
  x_eval = x;
  expr_eval(fofx_expr);
  expr_var_t *var = expr_var(&fofx_vars, "z", 1);
  return var->value;
}

double g_(vector_t x) {
  x_eval = x;
  expr_eval(gofx_expr);
  expr_var_t *var = expr_var(&gofx_vars, "z", 1);
  return var->value;
}


double g(vector_t x) {
  return g_(x) - constraint;
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
vector_t L_grad(vector_t v, double lambda, int minimize) {
  vector_t grad_f = f_grad(v);
  return vector_add(vector_mul(lambda, g_grad(v)), minimize? vector_inv(grad_f) : grad_f);
}

int solve_lagrange(int minimize, vector_t *x_out, double *lambda_out) {
  // simple gradient decent
  vector_t x = vector(1.0, 1.0);
  double lambda = 0.0;

  vector_t grad_L, grad_f;
  int n;
  for (n = 0; n < 3000; n++) {
    if (fabs(x.x) > 1e9 || fabs(x.y) > 1e9)
      return 0;
    grad_L = L_grad(x, lambda, minimize);
    x = vector_add(x, vector_mul( LEARNING_RATE, grad_L));

    lambda -= LEARNING_RATE * g(x);

    grad_f = f_grad(x);
    if (fabs(g(x)) < TOL && fabs(grad_f.x) < TOL && fabs(grad_f.y) < TOL) {
      break;
    }
  }
  printf(" + %d steps taken!\n", n);
  *x_out = x;
  *lambda_out = lambda;
  return 1;
}



int main() {
  setbuf( stdout, 0);
  printf("Hello world!\n");
  init();

  printf("enter constraint value:");
  scanf( "%lf", &constraint);
  fflush(stdin);
  printf("enter 0 to maximize, or 1 to minimize:");
  char c = getchar();
  fflush(stdin);
  int minimize = c != '0' ? 1 : 0;
  printf("finding %s solution...\n", minimize ? "minimal" : "maximal");
  double lambda;
  int found;
  vector_t x;
  found = solve_lagrange(minimize, &x, &lambda);
  if (found) {
    printf("f(x) & g(x).....: %.3lf & %.3lf\n", f(x), g_(x));
    printf("optimal x.......: (%.3lf, %.3lf)\n", x.x, x.y);
    printf("optimal lambda..: %.3lf\n", lambda);
    //export_objs(x);
  } else
    printf("%s solution d.n.e!\n", minimize ? "minimization" : "maximization");

  term();
  printf("Goodbye!\n");
  return 0;
}
