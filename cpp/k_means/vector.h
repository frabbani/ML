#pragma once

struct Vector2 {
  union {
    struct {
      double x, y;
    };
    double xy[2];
  };
  Vector2() {
    x = y = 0.0f;
  }
  Vector2(float x, float y) {
    this->x = x;
    this->y = y;
  }

  Vector2 operator +(const Vector2 &rhs) const {
    return Vector2(x + rhs.x, y + rhs.y);
  }
  Vector2 operator -(const Vector2 &rhs) const {
    return Vector2(x - rhs.x, y - rhs.y);
  }
  Vector2 operator *(float s) const {
    return Vector2(x * s, y * s);
  }

  double dot(const Vector2 &rhs) const {
    return x * rhs.x + y * rhs.y;
  }

  double length() {
    double s = dot(*this);
    if (s < 1e-16)
      return 0.0;
    return sqrt(s);
  }

  double lengthSq() {
    return dot(*this);
  }

  Vector2 normalized() const {
    Vector2 r;
    double s = dot(*this);
    if (s > 1e-16) {
      s = 1.0 / sqrt(s);
      return *this * s;
    }
    return r;
  }

  static Vector2 max(const Vector2 &lhs, const Vector2 &rhs) {
    double x = rhs.x > lhs.x ? rhs.x : lhs.x;
    double y = rhs.y > lhs.y ? rhs.y : lhs.y;
    return Vector2(x, y);
  }

  static Vector2 min(const Vector2 &lhs, const Vector2 &rhs) {
    double x = rhs.x < lhs.x ? rhs.x : lhs.x;
    double y = rhs.y < lhs.y ? rhs.y : lhs.y;
    return Vector2(x, y);
  }
};

Vector2 operator *(float s, const Vector2 &rhs) {
  return rhs * s;
}

