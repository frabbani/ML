#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <functional>

#include <mysdl2.h>

#include "svm.h"

#define DISP_W 256
#define DISP_H 256

using namespace sdl2;

SDL sdl;
SVM svm;
std::vector<SVM::DataPoint> dataPoints;
int drawType = 0;  // 0 - support vectors, 1 - evaluation, 2 - data points

Pixel24 *drawBuffer = nullptr;

void init() {
  printf("*** INIT ***\n");
  sdl2::Bitmap inputImage("input.bmp");
  printf("input image is %d x %d\n", inputImage.width(), inputImage.height());

  dataPoints.reserve(inputImage.width() * inputImage.height());
  auto is_white = [](Pixel24 px) {
    return px.r >= 128 && px.g >= 128 && px.b >= 128;
  };

  auto is_black = [](Pixel24 px) {
    return px.r < 32 && px.g < 32 && px.b < 32;
  };
  inputImage.lock();
  auto &pxs = inputImage.pixels;
  for (int y = 0; y < pxs.h; y++) {
    float t = 2.0 * float(y) / float(pxs.h - 1) - 1.0;
    for (int x = 0; x < pxs.w; x++) {
      float s = 2.0 * float(x) / float(pxs.w - 1) - 1.0;
      auto px = pxs.get24(x, y);
      if (is_white(*px) || is_black(*px))
        continue;
      SVM::DataPoint pt;
      pt.v = Vector2(s, t);
      pt.label = px->r > 128 ? 1 : -1;
      dataPoints.push_back(pt);
    }
  }

  int numReds = 0, numBlues = 0;
  for (auto pt : dataPoints) {
    if (pt.label > 0)
      numReds++;
    else
      numBlues++;
  }
  printf("%d total, %d red v. %d blue\n", int(dataPoints.size()), numReds, numBlues);
  void unLock();

  double error = svm.trainSMO(dataPoints);
  printf("SVM training error: %lf\n", error);

  drawBuffer = new Pixel24[DISP_H * DISP_H];

  for (int y = 0; y < DISP_H; y++) {
    float t = 2.0 * float(y) / float(DISP_H - 1) - 1.0;
    for (int x = 0; x < DISP_W; x++) {
      float s = 2.0 * float(x) / float(DISP_W - 1) - 1.0;
      SVM::DataPoint pt;
      pt.v = Vector2(s, t);
      pt.label = svm.predictLabel(pt);
      drawBuffer[y * DISP_W + x] = pt.label >= 0 ? Pixel24(255, 0, 0) : Pixel24(0, 0, 255);
    }
  }
}

void step() {
  if (sdl.keyDown('1'))
    drawType = 0;
  if (sdl.keyDown('2'))
    drawType = 1;
  if (sdl.keyDown('3'))
    drawType = 2;
}

void draw() {
  auto pixels = sdl.lock();

  if (drawType == 0) {
    for (int y = 0; y < pixels.h; y++)
      for (int x = 0; x < pixels.w; x++)
        pixels.plot(x, y, 0, 0, 0);

    for (auto pt : svm.supportVectors) {
      int x = pt.v.x * DISP_W / 2 + DISP_W / 2;
      int y = pt.v.y * DISP_H / 2 + DISP_H / 2;
      pixels.plot(x, y, pt.label == 1 ? Pixel24(255, 0, 0) : Pixel24(0, 0, 255));
    }
  }

  if (drawType == 1) {
    for (int y = 0; y < pixels.h; y++)
      for (int x = 0; x < pixels.w; x++)
        pixels.plot(x, y, 0, 0, 0);

    for (auto pt : dataPoints) {
      int x = pt.v.x * DISP_W / 2 + DISP_W / 2;
      int y = pt.v.y * DISP_H / 2 + DISP_H / 2;
      pixels.plot(x, y, pt.label == 1 ? Pixel24(255, 0, 0) : Pixel24(0, 0, 255));
    }
  }

  if (drawType == 2)
    for (int y = 0; y < DISP_H; y++)
      for (int x = 0; x < DISP_W; x++)
        pixels.plot(x, y, drawBuffer[y * DISP_W + x]);

}

void term() {
  printf("*** TERM ***\n");
  delete[] drawBuffer;

  printf("************\n");
}

int main(int argc, char *args[]) {
  setbuf( stdout, NULL);
  if (!sdl.init( DISP_W, DISP_H, false))
    return 0;

  setbuf( stdout, NULL);

  double drawTimeInSecs = 0.0;
  Uint32 drawCount = 0.0;
  double invFreq = 1.0 / sdl.getPerfFreq();

  auto start = sdl.getTicks();
  auto last = start;
  init();
  while (true) {
    sdl.pump();
    if (sdl.keyDown(SDLK_ESCAPE))
      break;

    auto now = sdl.getTicks();
    auto elapsed = now - last;

    if (elapsed >= 20) {
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
    printf("Average draw time: %f ms\n", (float) ((drawTimeInSecs * 1e3) / (double) drawCount));

  sdl.term();
  printf("goodbye!\n");
  return 0;
}
