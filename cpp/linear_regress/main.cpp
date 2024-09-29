#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <functional>

#include "mysdl2.h"	//https://github.com/frabbani/mysdl2
#include "vector.h"

extern "C" {
#include "mtwister.h"
}

MTRand mtRand;
double genRandomRange(double min, double max) {
  auto d = double(genRandLong(&mtRand)) / double(0xffffffff);
  return min + d * (max - min);
}

#define DISP_W 512
#define DISP_H 512

using namespace sdl2;

SDL sdl;

bool paused = false;

struct Line {
  double m = 0.0;
  double b = 0.0;
  double y(double x) {
    return m * x + b;
  }
  Vector2 norm() {
    return Vector2(-m, 1.0).normalized();
  }
  Vector2 randomPoint(double x, double dist) {
    dist *= genRandomRange(-1.0, +1.0);
    return Vector2(x, y(x)) + dist * norm();
  }
};

Line line, lineReg;
std::vector<Vector2> points;

void init() {
  printf("*** INIT ***\n");
  mtRand = seedRand(123);
  auto gen_random = [&](double min = 0.0, double max = 1.0) {
    // double s = double(rand()) / double(RAND_MAX - 1);
    // return min + s * ( max - min);
    return genRandomRange(min, max);
  };

  line.m = 0.333;
  line.b = DISP_H / 3;
  printf("y = %.3lfx + %.3lf\n", line.m, line.b);
  for (int i = 0; i < DISP_W / 2; i += 2) {
    double x = i * 2 + gen_random(-1, 1);
    points.push_back(line.randomPoint(x, 50.0));
  }
  printf("# of points generated: %zu\n", points.size());
  printf("************\n");
}

void step() {
  if (sdl.keyPress(SDLK_p))
    paused = !paused;
  if (paused)
    return;

  double N = double(points.size());
  double sumXY = 0.0;
  double sumX = 0.0;
  double sumY = 0.0;
  double sumXsq = 0.0;
  for (auto p : points) {
    sumXY += p.x * p.y;
    sumX += p.x;
    sumY += p.y;
    sumXsq += p.x * p.x;
  }
  lineReg.m = (N * sumXY - sumX * sumY) / (N * sumXsq - sumX * sumX);
  lineReg.b = (sumY - lineReg.m * sumX) / N;

}

void draw() {
  auto pixels = sdl.lock();
  auto draw_box = [&](Vector2 p, int w, int h, Pixel24 color) {
    h /= 2;
    w /= 2;
    for (int y = (int) p.y - h; y <= (int) p.y + h; y++)
      for (int x = (int) p.x - w; x <= (int) p.x + w; x++)
        pixels.plot(x, y, color);
  };

  auto draw_line = [&](double m, double b, Pixel24 color, bool dashed = false) {
    int dx = dashed ? 3 : 1;
    for (int x = 0; x < DISP_W; x += dx)
      pixels.plot(x, m * x + b, color);
  };

  pixels.inverted = true;
  for (int y = 0; y < pixels.h; y++)
    for (int x = 0; x < pixels.w; x++)
      pixels.plot(x, y, 195, 216, 222);

  for (auto p : points) {
    draw_box(p, 3, 3, Pixel24(255, 0, 0));
  }

  draw_line(line.m, line.b, Pixel24(64, 128, 0), true);
  draw_line(lineReg.m, lineReg.b, Pixel24(0, 0, 255));
}

void term() {
  printf("*** TERM ***\n");
  sdl.takeScreenshot();
  printf("************\n");
}

int main(int argc, char *args[]) {
  setbuf( stdout, NULL);

  if (!sdl.init( DISP_W, DISP_H, false, "line up!")) {
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
