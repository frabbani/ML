#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <functional>

#include <mysdl2.h>	//https://github.com/frabbani/mysdl2

#define DISP_W 512
#define DISP_H 512

#include "game.h"

using namespace sdl2;

SDL sdl;

Game g;
bool paused = true;

void init() {
  printf("*** INIT ***\n");
  g.trainChaser();
  g.runner = Vector2(0.0, 0.0);
  g.chaser = Vector2(-0.45, -0.45);
  paused = true;
  printf("************\n");
}

void step() {
  if (sdl.keyPress(SDLK_p))
    paused = !paused;
  if (paused)
    return;

  int x = 0, y = 0;
  if (sdl.keyDown(SDLK_LEFT))
    x--;
  if (sdl.keyDown(SDLK_RIGHT))
    x++;
  if (sdl.keyDown(SDLK_UP))
    y++;
  if (sdl.keyDown(SDLK_DOWN))
    y--;

  g.run(x, y, 0.01);
  g.chase();
}

void draw() {
  auto pixels = sdl.lock();
  auto draw_box = [&](Vector2 p, int w, int h, sdl2::Pixel24 color) {
    h /= 2;
    w /= 2;
    for (int y = (int) p.y - h; y <= (int) p.y + h; y++)
      for (int x = (int) p.x - w; x <= (int) p.x + w; x++)
        pixels.plot(x, y, color);
  };

  pixels.inverted = true;
  for (int y = 0; y < pixels.h; y++)
    for (int x = 0; x < pixels.w; x++)
      pixels.plot(x, y, 142, 195, 189);

  Vector2 p;
  Vector2 center = { DISP_W / 2, DISP_H / 2 };

  p.x = g.runner.x * DISP_W;
  p.y = g.runner.y * DISP_H;
  draw_box(center + p, 16, 16, Pixel24(123, 115, 170));

  p.x = g.chaser.x * DISP_W;
  p.y = g.chaser.y * DISP_H;
  draw_box(center + p, 24, 25, Pixel24(183, 18, 98));
}

void term() {
  printf("*** TERM ***\n");
  printf("************\n");
}

int main(int argc, char *args[]) {
  setbuf( stdout, NULL);

  if (!sdl.init( DISP_W, DISP_H, false, "Run Boy Run!")) {
    printf("SDL failed...?\n");
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
