#pragma once

#include <mysdl2.h>
using namespace sdl2;

#include "vector.h"
#include "neural.h"

#define PI 3.14159265358979

#define LERP( v, v2, a )  ( (1.0 - (a)) * (v) + (a) * (v2) )
#define CLAMP( v, l, h )  { v = v < (l) ? (l) : v > (h) ? (h) : v; }

bool isBlack(Pixel24 pixel) {
  return 5 >= pixel.r && 5 >= pixel.g && 5 >= pixel.b;
}

bool isRed(Pixel24 pixel) {
  return 250 <= pixel.r && 4 >= pixel.g && 4 >= pixel.b;
}

bool isGreen(Pixel24 pixel) {
  return 4 >= pixel.r && 250 <= pixel.g && 4 >= pixel.b;
}

Pixel24 blend(Pixel24 pixel, Pixel24 pixel2, float alpha) {
  CLAMP(alpha, 0.0, 1.0);
  float r = pixel.r + alpha * (pixel2.r - pixel.r);
  float g = pixel.g + alpha * (pixel2.g - pixel.g);
  float b = pixel.b + alpha * (pixel2.b - pixel.b);
  return Pixel24(r, g, b);
}

std::vector<Vector2> directions;

void initDirections() {
  directions.clear();
  for (int i = 0; i < 360; i += 15) {
    float angle = i;
    Vector2 v;
    v.x = cosf(angle * PI / 180.0f);
    v.y = sinf(angle * PI / 180.0f);
    directions.push_back(v);
  }
  printf("initDirections: %zu directions\n", directions.size());

}

struct Game {

  Game() {
    initDirections();
  }

  struct Map {
    int w = 0, h = 0;
    double downscale = 0.0f;
    std::vector<int> points;
    Vector2 origin;
    Vector2 target;
    std::vector<Pixel24> colors;
    std::vector<int> visits;
    void init(std::string_view file) {
      Bitmap bitmap(file.data());
      w = bitmap.width();
      h = bitmap.height();
      points = std::vector<int>(w * h);
      colors = std::vector<Pixel24>(w * h);
      visits = std::vector<int>(w * h);

      bitmap.lock();
      auto &pixels = bitmap.pixels;
      int tot = 0, tot2 = 0;
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
          auto pix = *pixels.get24(x, y);
          points[y * w + x] = isBlack(pix) ? 0 : 1;
          if (isRed(pix)) {
            tot++;
            origin = origin + Vector2(x, y);
          }
          if (isGreen(pix)) {
            tot2++;
            target = target + Vector2(x, y);
          }
          colors[y * w + x] = pix;
        }
      bitmap.unlock();
      downscale = 1.0 / sqrt(w * w + h * h);

      origin = origin * (1.0 / tot);
      target = target * (1.0 / tot2);

      printf("origin.: { %d, %d }\n", int(origin.x), int(origin.y));
      printf("target.: { %d, %d }\n", int(target.x), int(target.y));

    }

    bool oob(int x, int y) {
      if (x < 0 || x >= w || y < 0 || y >= h)
        return true;
      return points[y * w + x] == 0;
    }

    int& get(int x, int y) {
      CLAMP(x, 0, w - 1);
      CLAMP(y, 0, h - 1);
      return points[y * w + x];
    }

    float traceLine(Vector2 p, Vector2 p2) {
      // modified Bresenham
      int dx = abs(p2.x - p.x);
      int dy = abs(p2.y - p.y);
      int dist = dx > dy ? dx : dy;
      if (0 == dist)
        return 1.0f;

      float alpha = 0.0f, alpha2 = 0.0f;
      for (int d = 0; d <= dist; d++) {
        alpha = alpha2;
        alpha2 = float(d) / float(dist);
        Vector2 u = LERP(p, p2, alpha2);

        int x = (int) u.x;
        int x2 = u.x > x ? x + 1 : x;
        int y = (int) u.y;
        int y2 = u.y > y ? y + 1 : y;

        if (oob(x, y) || oob(x2, y) || oob(x, y2) || oob(x2, y2))
          return alpha;
      }
      return 1.0;
    }

    Pixel24 drawColor(int x, int y) {
      CLAMP(x, 0, w - 1);
      CLAMP(y, 0, h - 1);
      return colors[y * w + x];
    }

    void visit(int x, int y) {
      CLAMP(x, 0, w - 1);
      CLAMP(y, 0, h - 1);
      visits[y * w + x] = 1;
    }
    bool visited(int x, int y) {
      CLAMP(x, 0, w - 1);
      CLAMP(y, 0, h - 1);
      return visits[y * w + x] == 1;
    }

    void clearVisited() {
      for (auto &v : visits)
        v = 0;
    }
  };

  struct Agent {
    static constexpr double ewmaWeight = 0.01;
    static constexpr double maxDist = 16.0;
    static constexpr double lingerDist = 4.0;
    Vector2 p;
    Vector2 c;
    int action = 0;
    Map *map;
    void init(Map &map_) {
      this->map = &map_;
      reset();
    }

    void reset() {
      if (map) {
        action = rand() % directions.size();
        p = map->origin;
        c = p - directions[action] * (3.0 * lingerDist);
      }
    }

    Vector2 d() {
      return directions[action];
    }

    void step() {
      p = p + d();
      map->visit(p.x, p.y);
      auto d = p - c;
      if (d.dot(d) >= (lingerDist * lingerDist))
        c = LERP(c, p, ewmaWeight);
    }

    double reward() {
      double exitReward = 0.0;
      double lingerPenalty = 0.0;
      double hitPenalty = 0.0;
      double projection = 0.0;

      auto dir = map->target - p;  // point towards exit
      double lenSq = dir.dot(dir);
      if (lenSq < maxDist * maxDist)
        exitReward = 1;
      else {
        double len = sqrt(lenSq);
        exitReward = (1.0 - len * map->downscale);
        projection = dir.normalized().dot(directions[action]);
        exitReward *= projection;
      };

      dir = c - p;  // point towards center
      lenSq = dir.dot(dir);
      if (lenSq <= lingerDist * lingerDist) {
        lingerPenalty = 1.0;
        projection = dir.normalized().dot(directions[action]);
        CLAMP(projection, 0.0, 1.0);
        lingerPenalty *= projection;
      }

      hitPenalty = (1.0 - traceLine());

      double reward = exitReward - 0.3 * lingerPenalty - 0.1 * hitPenalty;
      //printf("cumulative reward %lf\n", reward);
      //printf(" * exit..: %lf\n", exitReward);
      //printf(" * linger: %lf\n", lingerPenalty);
      //printf(" * hit...: %lf\n", hitPenalty);
      CLAMP(reward, -1.0, 1.0);
      return reward;

    }

    bool traceHit() {
      Vector2 p2 = p + maxDist * d();
      float alpha = map->traceLine(p, p2);
      return alpha < 1.0;
    }

    float traceLine() {
      Vector2 p2 = p + maxDist * d();
      return map->traceLine(p, p2);
    }

    void nnSetup(NN_neural_network_t *nn) {
      // normalize inputs
      nn->input[0] = map->target.x * map->downscale;
      nn->input[1] = map->target.y * map->downscale;
      nn->input[2] = p.x * map->downscale;
      nn->input[3] = p.y * map->downscale;
      nn->input[4] = c.x * map->downscale;
      nn->input[5] = c.y * map->downscale;
      for (size_t i = 0; i < directions.size(); i++) {
        nn->input[6 + i] = traceLine();
      }
    }

    Pixel24 color() {
      return Pixel24(255, 0, 255);
    }

  };

  NN_neural_network_t *nn = nullptr;
  std::vector<double> currQ;
  std::vector<double> nextQ;

  double learn = 0.07;  // neural network learning rate
  double gamma = 0.7;   // q learning discount factor
  double epsilon = 0.66;  // epsilon greedy epsilon value
  double alpha = 0.3;  // Bellman learning rate

  Map map;
  Agent explorer;

  void init() {

    map.init("map.bmp");

    explorer.init(map);

    NN_info_t info;
    info.hidden_layers_size = 2;

    info.activation = NN_relu;
    info.input_size = 2 + 2 + 2 + directions.size();  //target, position, ewma + traces
    info.output_size = directions.size();  //Q calues

    nn = new NN_neural_network_t;
    for (int i = 0; i < info.hidden_layers_size; i++)
      info.neurons_per[i] = 10;
    NN_init_neural_network(nn, &info);  // 1 hidden layer with 4 neurons

    explorer.nnSetup(nn);

    currQ = std::vector<double>(directions.size());
    nextQ = std::vector<double>(directions.size());
  }

  int eGreedy() {
    int qSize = (int) directions.size();

    if (NN_random(1.0, 0.0) <= epsilon)
      // Exploration: select a random action
      return rand() % qSize;

    // Exploitation: select the action with the highest Q-value
    int best = 0;
    double qMax = currQ[0];
    for (int i = 1; i < qSize; i++) {
      float q = currQ[i];
      if (q > qMax) {
        qMax = q;
        best = i;
      }
    }
    return best;
  }

  void explore() {
    int qSize = (int) directions.size();  // same nn output/target size, currQ size && nextQ size;

    auto set_q = [&](std::vector<double> &q) {
      for (int i = 0; i < qSize; i++)
        q[i] = nn->output_layer.neurons[i].value;
    };

    auto max_q = [&](std::vector<double> &q) {
      double max = q[0];
      for (int i = 0; i < qSize; i++)
        if (q[i] > max)
          max = q[i];
      return max;
    };

    explorer.nnSetup(nn);
    NN_forward_propagate(nn);
    set_q(currQ);

    explorer.action = eGreedy();
    if (!explorer.traceHit())
      explorer.step();

    explorer.nnSetup(nn);
    NN_forward_propagate(nn);
    set_q(nextQ);

    double actionTarget = explorer.reward() + gamma * max_q(nextQ);
    for (int i = 0; i < qSize; i++)
      nn->target[i] =
          i == explorer.action ?
              currQ[i] + alpha * (actionTarget - currQ[i]) : currQ[i];
    NN_backward_propagate(nn, learn);

  }

  ~Game() {
    delete nn;
  }

};
