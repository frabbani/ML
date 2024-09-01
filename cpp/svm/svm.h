#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "vector.h"
/*
 2-FEATURE/ 2-LABEL STATE VECTOR MACHINE
 */

struct SVM {
  static constexpr int C = 25;
  static constexpr double TOL = 1e-5;
  static constexpr int N = 500;

  struct DataPoint {
    Vector2 v;
    int label;
  };

  std::vector<DataPoint> supportVectors;
  std::vector<double> supportAlphas;  // Lagrange multipliers
  int bias;

  double rbfKernel(Vector2 v, Vector2 v2) {
    const float sigma = 3.0;
    Vector2 d = v2 - v;
    return exp(-d.dot(d) / (2 * sigma * sigma));
  }

  double polyKernel(Vector2 v, Vector2 v2) {
    float s = v.dot(v2) + 0.5;
    s *= s;
    return s * s;
  }

  double kernel(Vector2 v, Vector2 v2) {
#if 0
    return rbfKernel( v, v2 );
  #else
    return polyKernel(v, v2);
#endif
  }

  // Prediction function
  int predictLabel(const DataPoint &testPoint) {
    double result = bias;
    for (size_t i = 0; i < supportVectors.size(); i++) {
      result += supportAlphas[i] * supportVectors[i].label * kernel(supportVectors[i].v, testPoint.v);
    }
    return (result >= 0) ? 1 : -1;
  }

  double trainSMO(std::vector<DataPoint> data) {
    bias = 0;
    auto alphas = std::vector<double>(data.size());
    for (size_t i = 0; i < data.size(); i++) {
      alphas[i] = 0;
    }

    int numPoints = int(data.size());
    int iters = 0;
    while (iters < N) {
      bool modified = false;

      for (int i = 0; i < numPoints; i++) {
        // Compute error for the i-th data point
        double Ei = bias;
        for (int j = 0; j < numPoints; j++) {
          if (alphas[j] > 0) {
            Ei += alphas[j] * data[j].label * kernel(data[i].v, data[j].v);
          }
        }
        Ei -= data[i].label;

        if ((data[i].label * Ei < -TOL && alphas[i] < C) || (data[i].label * Ei > TOL && alphas[i] > 0)) {
          // Select a random point where j != i
          int j = rand() % numPoints;
          while (j == i)
            j = rand() % numPoints;

          // Compute error for the j-th data point
          double Ej = bias;
          for (int k = 0; k < numPoints; k++) {
            if (alphas[k] > 0) {
              Ej += alphas[k] * data[k].label * kernel(data[j].v, data[k].v);
            }
          }
          Ej -= data[j].label;

          // Retrieve the current values of the Lagrange multipliers for the selected data points
          double iAlpha = alphas[i];
          double jAlpha = alphas[j];

          // Lagrangian multipliers constraints
          // a_i >= 0
          // a_j >= 0
          // a_i <= C
          // a_j <= C

          // Compute bounds
          double L, H;
          if (data[i].label != data[j].label) {
            L = fmax(0, jAlpha - iAlpha);
            H = fmin(C, C + jAlpha - iAlpha);
          } else {
            L = fmax(0, iAlpha + jAlpha - C);
            H = fmin(C, iAlpha + jAlpha);
          }

          // Skip iteration if the lower and upper bounds are the same, meaning
          // there is no valid range for updating j-alpha (and thus no optimization can be performed)
          if (L == H)
            continue;

          // Compute eta as part of the quadratic component in SVM optimization
          //  * eta = 2 * K(x_i, x_j) - K(x_i, x_i) - K(x_j, x_j)
          //
          // Where:
          // K(x_i, x_j) is the kernel function applied to data points i and j
          // K(x_i, x_i) is the kernel function applied to data point i (diagonal element)
          // K(x_j, x_j) is the kernel function applied to data point j (diagonal element)
          //
          // eta is used to determine if the optimization step should be taken
          // If eta >= 0, no improvement is possible, so we skip the update for these points

          double eta = 2 * kernel(data[i].v, data[j].v) - kernel(data[i].v, data[i].v) - kernel(data[j].v, data[j].v);
          if (eta >= 0)
            continue;

          // Update j-alpha
          alphas[j] -= data[j].label * (Ei - Ej) / eta;
          if (alphas[j] > H)
            alphas[j] = H;
          else if (alphas[j] < L)
            alphas[j] = L;

          if (fabs(alphas[j] - jAlpha) < TOL)
            continue;

          // Update i-alpha
          alphas[i] += data[i].label * data[j].label * (jAlpha - alphas[j]);

          // Update bias
          double b1 = bias - Ei - data[i].label * (alphas[i] - iAlpha) * kernel(data[i].v, data[i].v) - data[j].label * (alphas[j] - jAlpha) * kernel(data[i].v, data[j].v);
          double b2 = bias - Ej - data[i].label * (alphas[i] - iAlpha) * kernel(data[i].v, data[j].v) - data[j].label * (alphas[j] - jAlpha) * kernel(data[j].v, data[j].v);

          if (0 < alphas[i] && alphas[i] < C)
            bias = b1;
          else if (0 < alphas[j] && alphas[j] < C)
            bias = b2;
          else
            bias = (b1 + b2) / 2;

          modified = true;
        }
      }

      if (!modified)
        break;
      iters++;
    }

    supportVectors.clear();
    supportAlphas.clear();
    for (int i = 0; i < numPoints; i++)
      if (alphas[i] > TOL) {
        supportVectors.push_back(data[i]);
        supportAlphas.push_back(alphas[i]);
      }

    int correct = 0;
    for (auto point : data) {
      int prediction = predictLabel(point);
      if (prediction == point.label)
        correct++;
    }
    return (double) (numPoints - correct) / (double) numPoints;
  }

};

