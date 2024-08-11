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

bool paused = false;

void init() {
  printf("*** INIT ***\n");
  g.init();
  printf("************\n");
}

void step() {
  if (sdl.keyPress(SDLK_p))
    paused = !paused;
  if (paused)
    return;
  if( sdl.keyPress(SDLK_r)){
    g.explorer.reset();
  }

  if( sdl.keyPress(SDLK_c)){
    g.map.clearVisited();
  }
  g.explore();
}

void draw() {
  auto pixels = sdl.lock();
  for (int y = 0; y < pixels.h; y++)
    for (int x = 0; x < pixels.w; x++) {
      if( g.map.visited(x, y) )
        pixels.plot(x, y, 128, 0, 64 );
      else
        pixels.plot(x, y, g.map.drawColor(x, y));
    }

  auto draw_faded_box = [&](Vector2 p, int d, Pixel24 color, float alpha) {
    for (int y = (int) p.y - d; y <= (int) p.y + d; y++)
      for (int x = (int) p.x - d; x <= (int) p.x + d; x++) {
        auto c = blend(g.map.drawColor(x, y), color, alpha);
        pixels.plot(x, y, c.r, c.g, c.b);
      }
  };

  draw_faded_box(g.explorer.p, 4, g.explorer.color(), 1.0);
  draw_faded_box(g.explorer.c, 3, g.explorer.color(), 0.25);

}

void term() {
  printf("*** TERM ***\n");
  printf("************\n");
}

int main(int argc, char *args[]) {
  setbuf( stdout, NULL);

  if (!sdl.init( DISP_W, DISP_H, false, "Dora the Explorer!")) {
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
