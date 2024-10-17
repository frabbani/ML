#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <functional>

#include "mysdl2.h"	//https://github.com/frabbani/mysdl2
#include "vector.h"
#include "logistic_regress.h"

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
bool paused = true;

SDL sdl;
int drawType = 1;

struct LinearClassifier {
  double m, b;

 public:
  double fofx(double x) {
    return m * x + b;
  }

  LinearClassifier(double m_, double b_)
      :
      m(m_),
      b(b_) {
  }

  int classify(Vector2 v) {
    double y = fofx(v.x);
    return v.y > y ? 1 : 0;
  }
};

Vector2 jitter(Vector2 v, double xMin, double xMax, double yMin, double yMax) {
  double x = v.x + genRandomRange(xMin, xMax);
  double y = v.y + genRandomRange(yMin, yMax);
  return Vector2(x, y);
}

std::vector<Point> points;
LogisticRegression logReg;
std::vector<Point> testPoints;
LinearClassifier classifier(0.33, DISP_H / 3);
int iterations = 0;
constexpr int maxIterations = 6000;
bool drawSource = false;

void drawText(const std::string &text, int x, int y) {
  auto glyphs = sdl.fonts["night"]->glyphs;
  sdl2::Rect rect;
  rect.x = x;
  rect.y = y;
  for (auto c : text) {
    auto glyph = glyphs[c - ' '];
    if (glyph) {
      rect.w = glyph->w;
      rect.h = glyph->h;
      SDL_BlitSurface(glyph, nullptr, sdl.surf, &rect);
      rect.x += glyph->w;
    }
  }
}

void drawText(Pixels &pixels, const std::string &text, int x, int y,
              Pixel24 color) {
  sdl.fonts["night machine"]->render( pixels, x, y, text, color );

}

void init() {
  printf("*** INIT ***\n");
  mtRand = seedRand(123);

  for (int i = 0; i < DISP_W; i++) {
    Vector2 v(double(i), classifier.fofx(double(i)));
    v = jitter(v, -1.0, +1.0, -150.0, +150.0);
    Point pt;
    pt.v = v;
    pt.label = classifier.classify(pt.v);
    points.push_back(pt);
  }
  printf("# of points generated: %zu\n", points.size());

  logReg.setBounds(points);

  testPoints.clear();
  for (int y = 0; y < DISP_H; y++) {
    for (int x = 0; x < DISP_W; x++) {
      Point pt;
      pt.v = Vector2(x, y);
      pt.label = 0;
      testPoints.push_back(pt);
    }
  }

  sdl.loadFont("night machine", "Night Machine.otf", 24);

  printf("************\n");
}

void step() {
  if (sdl.keyPress(SDLK_p))
    paused = !paused;

  if (sdl.keyPress(SDLK_1))
    drawType = 1;
  if (sdl.keyPress(SDLK_2))
    drawType = 2;

  if (paused)
    return;

  if (iterations < maxIterations) {
    iterations += 5;
    logReg.trainingStep(points, 5);
  }

  for (auto &pt : testPoints)
    pt.label = logReg.predictLabel(pt.v);

  if (sdl.keyPress('\''))
    sdl.takeScreenshot();
}

void draw() {
  auto pixels = sdl.lock();

  Pixel24 bgColor = Pixel24(28, 28, 46);
  Pixel24 labelColor = Pixel24(44, 135, 156);
  Pixel24 labelColor2 = Pixel24(37, 148, 59);
  Pixel24 classColor = Pixel24(220, 180, 5);
  Pixel24 textColor = Pixel24(230, 160, 55);
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

  if (drawSource) {
    for (int x = 0; x < DISP_W; x += 8) {
      int y = int(classifier.fofx(double(x)));
      pixels.plot(x, y, classColor);
    }
  }

  if (drawType == 1) {
    for (auto pt : points)
      draw_box(pt.v, 3, 3, pt.label > 0 ? labelColor : labelColor2);
    for (auto pt : points) {
      if (logReg.predictLabel(pt.v) != pt.label)
        draw_box(pt.v, 3, 3, Pixel24(255, 0, 0));
    }
  } else {
    for (auto pt : testPoints)
      pixels.plot(pt.v.x, pt.v.y, pt.label > 0 ? labelColor : labelColor2);
  }

  std::string text = "iterations:  " + std::to_string(iterations);
  drawText(pixels, text, 10, 10, textColor);
}

void term() {
  printf("*** TERM ***\n");
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
