#pragma once

#include "vector.h"

struct Point {
  Vector2 v;
  int label;
};

class LogisticRegression {
  double beta0 = 0.0, beta1 = 0.0, beta2 = 0.0;
  Vector2 min, max;

  Vector2 norm(Vector2 v) {
    for (int i = 0; i < 2; i++) {
      v.xy[i] -= min.xy[i];
      double d = max.xy[i] - min.xy[i];
      v.xy[i] /= (d > 1e-5 ? d : 1.0);
    }
    return v;
  }

  double logistic(double z) {
    return 1.0 / (1.0 + exp(-z));
  }

 public:

  void setBounds(const std::vector<Point> &points) {
    min = max = points[0].v;
    for (auto pt : points) {
      min = Vector2::min(min, pt.v);
      max = Vector2::max(max, pt.v);
    }
  }

  void trainingStep(const std::vector<Point> &points, int epochs = 25) {
    double s = 1.0 / double(points.size());
    for (int k = 0; k < epochs; k++) {
      double grad0 = 0.0, grad1 = 0.0, grad2 = 0.0;
      for (auto pt : points) {
        double pred = predict(pt.v);
        double err = double(pt.label) - pred;
        grad0 += err;
        grad1 += err * pt.v.x;
        grad2 += err * pt.v.y;
      }
      grad0 *= s;
      grad1 *= s;
      grad2 *= s;
      beta0 += grad0;
      beta1 += grad1;
      beta2 += grad2;
    }
  }

  double predict(Vector2 v) {
    v = norm(v);
    return logistic(beta0 + beta1 * v.x + beta2 * v.y);
  }

  int predictLabel(Vector2 v) {
    return predict(v) > 0.5 ? 1.0 : 0.0;
  }

  void train(const std::vector<Point> &points, double learningRate = 0.03,
             int epochs = 25000) {
    setBounds(points);
    double s = 1.0 / double(points.size());
    for (int k = 0; k < epochs; k++) {
      double grad0 = 0.0, grad1 = 0.0, grad2 = 0.0;
      for (auto pt : points) {
        double pred = predict(pt.v);
        double err = double(pt.label) - pred;
        grad0 += err;
        grad1 += err * pt.v.x;
        grad2 += err * pt.v.y;
      }
      grad0 *= s;
      grad1 *= s;
      grad2 *= s;
      beta0 += learningRate * grad0;
      beta1 += learningRate * grad1;
      beta2 += learningRate * grad2;
    }
  }
};
