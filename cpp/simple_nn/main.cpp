#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <functional>

#include "mysdl2.h"	//https://github.com/frabbani/mysdl2
#include "nn.h"

#define DISP_W 512
#define DISP_H 512

using namespace sdl2;

double random_range(double min, double max) {
  union {
    unsigned int v;
    unsigned char arr[2];
  } u;
  u.arr[0] = rand() % 256;
  u.arr[1] = rand() % 256;
  u.arr[2] = rand() % 256;
  u.arr[3] = rand() % 256;

  double s = (double) u.v / (double) 0xFFFFFFFE;
  return min + s * (max - min);
}

SDL sdl;
int drawType = 1;
bool paused = true;
std::shared_ptr<Font> font = nullptr;
std::shared_ptr<Font> smallFont = nullptr;

void renderText(Pixels &bg, int x, int y, Pixel24 color, const char *format,
                ...) {
  if (!font)
    return;
  char text[256];
  va_list args;
  va_start(args, format);
  vsnprintf(text, sizeof(text), format, args);  // Use vsnprintf for safety
  va_end(args);
  font->render(bg, x, y, text, color);
}

void renderTextSmall(Pixels &bg, int x, int y, Pixel24 color,
                     const char *format, ...) {
  if (!smallFont)
    return;

  char text[256];
  va_list args;
  va_start(args, format);
  vsnprintf(text, sizeof(text), format, args);  // Use vsnprintf for safety
  va_end(args);
  smallFont->render(bg, x, y, text, color);
}

int iterationsCount = 0;
const nn::Activation::Type activationType = nn::Activation::Type::Sigmoid;
nn::NeuralNetwork neuralNetwork(activationType);

double getTarget(double input) {
  return exp(-5.0 * input);
}

void init() {
  printf("*** INIT ***\n");

  TTF_Init();
  font = std::make_shared<Font>("Armwarmer.otf", 22);
  smallFont = std::make_shared<Font>("Thyssen J Italic.TTF", 22);

//  double &input = neuralNetwork.input;
//  double &output = neuralNetwork.prediction;

//  FILE *fp = fopen("output.csv", "w");
//  fprintf(fp, "x,y\n");
//  for (int i = 1; i <= 100; i++) {
//    input = (double) i / 100.0;
//    neuralNetwork.propagateForward();
//    printf("output v. target: %.4lf v. %.4lf\n", output, getTarget(input));
//    fprintf(fp, "%.4lf,%.4lf\n", input, output);
//  }
//  fclose(fp);

  printf("************\n");
}

void step() {
  if (sdl.keyPress('p'))
    paused = !paused;
  if (paused)
    return;

  if (iterationsCount < nn::numEpochs) {
    for (int i = 0; i < 100; i++) {
      iterationsCount++;
      double &input = neuralNetwork.input;
      input = random_range(0.0, 1.0);
      double target = getTarget(input);
      neuralNetwork.propagateForward();
      neuralNetwork.propagateBack(target);
    }
  }
}

void draw() {
  auto bg = sdl.lock();

  Pixel24 bgColor(10, 10, 25);
  Pixel24 targetColor(0, 128, 0);
  Pixel24 predictedColor(255, 120, 0);
  Pixel24 oobColor(100, 50, 0);

  int yBottom = DISP_W / 2 - DISP_W / 8;
  int yTop = yBottom + DISP_W / 2;

  bg.inverted = true;
  for (int y = 0; y < bg.h; y++)
    for (int x = 0; x < bg.w; x++)
      bg.plot(x, y, bgColor);

  for (int x = 0; x < DISP_W; x += 3) {
    bg.plot(x, yTop, 255, 255, 255);
    bg.plot(x, yBottom, 255, 255, 255);
  }

  for (int i = 0; i < DISP_W; i++) {
    double x = double(i) / double(DISP_W - 1);
    double y = getTarget(x) * DISP_H * 0.5;
    bg.plot(i, y + yBottom, targetColor);

    neuralNetwork.input = x;
    neuralNetwork.propagateForward();
    y = neuralNetwork.prediction * DISP_H * 0.5;
    bg.plot(i, y + yBottom, y < 0 ? oobColor : predictedColor);
  }

  renderTextSmall(bg, 95, DISP_H - 38, targetColor, "%s", "-5x");
  renderTextSmall(bg, 30, DISP_H - 50, targetColor, "%s", "y  =  e");
  //renderText( bg, 25, 100, Pixel24(50, 170, 255), "%s", "ACTIVATION:  Leaky ReLU" );
  renderText(
      bg,
      25,
      100,
      Pixel24(50, 170, 255),
      "%s: %s",
      "ACTIVATION",
      activationType == nn::Activation::Type::LeakyReLU ?
          "Leaky ReLU" : "Sigmoid");
  renderText(bg, 25, 65, Pixel24(50, 170, 255), "%s %d", "ITERATIONS:",
             iterationsCount);
}

void term() {
  printf("*** TERM ***\n");
  printf("************\n");
}

int main(int argc, char *args[]) {
  setbuf( stdout, NULL);

  if (!sdl.init( DISP_W, DISP_H, false, "what are you?")) {
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
