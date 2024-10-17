#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <functional>

#include "mysdl2.h"	//https://github.com/frabbani/mysdl2
#include "vector.h"

#define DISP_W 512
#define DISP_H 512

extern "C" {
#include "mtwister.h"
}

using namespace sdl2;

SDL sdl;
int drawType = 1;
bool paused = true;

#define CLAMP( v, a, b ) { v = v < (a) ? (a) : v > (b) ? (b) : v; }

MTRand mtRand;
double genRandom(double min = 0.0, double max = 1.0) {
  auto d = double(genRandLong(&mtRand)) / double(0xffffffff);
  return min + d * (max - min);
}

int genRand(int min, int max) {
  Uint32 d = abs(max - min) + 1;
  return int(genRandLong(&mtRand) % d) - min;
}

double genRandomNormal(double sigma, double mu = 0.0) {
  // Box-Muller transform

  double u1 = genRandom();
  double u2 = genRandom();
  double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);

  // Scale and shift
  return mu + z0 * sigma;
}

constexpr int numClusters = 5;
constexpr double learningRate = 0.3;

std::array<Pixel24, numClusters> labelColors;
Pixel24 unlabeledColor(0, 64, 32);

Pixel24 labelColor( int label ){
  if( label < 0 || label > numClusters )
    return unlabeledColor;
  return labelColors[label];

}

struct Point {
  Vector2 o;        // origin
  int label = -1;   // cluster no.
};

std::vector<Point> points;
std::array<Vector2, numClusters> origins;
std::array<Vector2, numClusters> centroids;
std::array<std::vector<Vector2>, numClusters> clusters;

void reset() {
  drawType = 0;
  for (int i = 0; i < numClusters; i++) {
    origins[i].x = genRandom(0.1 * DISP_W, 0.9 * DISP_W);
    origins[i].y = genRandom(0.1 * DISP_H, 0.9 * DISP_H);
  }

  points.clear();
  for (auto o : origins) {
    int numPoints = int(genRandom(200, 300));
    int sigma = genRandom(30, 60);
    for (int i = 0; i < numPoints; i++) {
      Vector2 v;
      v.x = o.x + genRandomNormal(sigma);
      v.y = o.y + genRandomNormal(sigma);
      points.push_back(Point { v, -1 });
    }

    // clear the centroids
    for (auto &c : centroids) {
      int i = genRand(0, points.size() - 1);
      auto pt = points[i];
      c.x = pt.o.x;
      c.y = pt.o.y;
    }
  }
}

void init() {
  printf("*** INIT ***\n");
  mtRand = seedRand(123);

  reset();
  for (auto &color : labelColors) {
    color.r = (Uint8) genRandom(25, 250);
    color.g = (Uint8) genRandom(25, 250);
    color.b = (Uint8) genRandom(25, 250);
  }

  printf("************\n");
}

void step() {
  if (sdl.keyPress('p'))
    paused = !paused;
  if (paused)
    return;
  if (sdl.keyPress('r'))
    reset();

  if (sdl.keyPress('\''))
    sdl.takeScreenshot();

  for (auto &p : points) {
    p.label = 0;
    double min = (p.o - centroids[0]).lengthSq();

    for (int index = 1; index < numClusters; index++) {
      double minIndex = (p.o - centroids[index]).lengthSq();
      if (minIndex < min) {
        min = minIndex;
        p.label = index;
      }
    }
  }

  for (int i = 0; i < numClusters; i++)
    clusters[i].clear();
  for (auto p : points) {
    if (p.label >= 0)
      clusters[p.label].push_back(p.o);
  }

  auto centroid_location = [](const std::vector<Vector2> &list) {
    double s = 1.0 / double(list.size());
    Vector2 c;
    for (auto p : list)
      c = c + s * p;
    return c;
  };

  bool converged = true;
  for (int i = 0; i < numClusters; i++) {
    if (clusters[i].size() > 1) {
      auto oldCentroid = centroids[i];
      auto newCentroid = centroid_location(clusters[i]);

      centroids[i] = (1.0 - learningRate) * centroids[i]
          + learningRate * newCentroid;

      if ((oldCentroid - newCentroid).lengthSq() > 1e-4)
        converged = false;
    } else {
      converged = false;
      centroids[i] = points[rand() % points.size()].o;
    }
  }

  drawType = int(converged);
}

void draw() {
  auto pixels = sdl.lock();

  Pixel24 bgColor(10, 10, 25);
  Pixel24 pointColor(0, 64, 32);
  Pixel24 originColor(128, 128, 255);
  Pixel24 centroidColor(128, 255, 255);

  auto alpha_blend = [](Pixel24 color, Pixel24 color2, float alpha) {
    float r = color.r + alpha * (color2.r - color.r);
    float g = color.g + alpha * (color2.g - color.g);
    float b = color.b + alpha * (color2.b - color.b);
    CLAMP(r, 0.0f, 255.0f);
    CLAMP(g, 0.0f, 255.0f);
    CLAMP(b, 0.0f, 255.0f);
    return Pixel24(r, g, b);
  };

  auto draw_x = [&](Vector2 p, Pixel24 color) {
    /* @formatter:on */
    const int icon[9][9] = { { 0, 0, 0, 0, 1, 0, 0, 0, 0 }, { 0, 0, 0, 1, 1, 1,
        0, 0, 0 }, { 0, 0, 1, 1, 0, 1, 1, 0, 0 }, { 0, 1, 1, 0, 0, 0, 1, 1, 0 },
        { 1, 1, 0, 0, 0, 0, 0, 1, 1 }, { 0, 1, 1, 0, 0, 0, 1, 1, 0 }, { 0, 0, 1,
            1, 0, 1, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1, 0, 0, 0 }, { 0, 0, 0, 0, 1,
            0, 0, 0, 0 }, };
    /* @formatter:on */

    for (int y = 0; y < 9; y++)
      for (int x = 0; x < 9; x++)
        if (icon[y][x])
          pixels.plot((int) p.x + x - 3, (int) p.y + y - 2, color);
  };

  auto draw_box = [&](Vector2 p, int w, int h, Pixel24 color) {
    h /= 2;
    w /= 2;
    for (int y = (int) p.y - h; y <= (int) p.y + h; y++)
      for (int x = (int) p.x - w; x <= (int) p.x + w; x++)
        pixels.plot(x, y, color);
  };

  pixels.inverted = true;
  for (int y = 0; y < pixels.h; y++)
    for (int x = 0; x < pixels.w; x++)
      pixels.plot(x, y, bgColor);

  for (auto p : points) {
    draw_box(p.o, 3, 3,
             alpha_blend(pointColor, labelColor(p.label), drawType ? 0.5 : 0.0));
  }

//  for (int i = 0; i < int(origins.size()); i++)
//    draw_x(origins[i], originColor);

  for (int i = 0; i < int(centroids.size()); i++)
    draw_x(centroids[i], centroidColor);

}

void term() {
  printf("*** TERM ***\n");
  printf("************\n");
}

int main(int argc, char *args[]) {
  setbuf( stdout, NULL);

  if (!sdl.init( DISP_W, DISP_H, false, "come at me, bro!")) {
    return 0;
  }

  double drawTimeInSecs = 0.0;
  Uint32 drawCount = 0.0;
  double invFreq = 1.0 / sdl.getPerfFreq();

  auto start = sdl.getTicks();
  auto last = start;
  init();

  while (true) {
    auto now = sdl.getTicks();
    auto elapsed = now - last;
    if (sdl.keyDown(SDLK_ESCAPE))
      break;
    if (elapsed >= 20) {
      sdl.pump();
      step();
      last = (now / 20) * 20;
    }
    Uint64 beginCount = sdl.getPerfCounter();
    draw();
    drawCount++;
    sdl.swap();
    drawTimeInSecs += (double) (sdl.getPerfCounter() - beginCount) * invFreq;
  }
  term();
  if (drawCount)
    printf("Average draw time: %f ms\n",
           (float) ((drawTimeInSecs * 1e3) / (double) drawCount));

  sdl.term();

  printf("goodbye!\n");
  return 0;
}
