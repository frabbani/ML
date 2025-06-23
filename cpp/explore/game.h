#pragma once

#include "mysdl2.h"
using namespace sdl2;

#include "vector.h"
#include "reinforce.h"
#include <optional>

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
  for (int i = 0; i < 360; i += 30) {
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
  int ticks = 0;

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
      downscale = 1.0 / radius();

      origin = origin * (1.0 / tot);
      target = target * (1.0 / tot2);

      printf("origin.: { %d, %d }\n", int(origin.x), int(origin.y));
      printf("target.: { %d, %d }\n", int(target.x), int(target.y));

    }
    double radiusSq() {
      return double(w * w + h * h);
    }
    double radius() {
      return sqrt(radiusSq());
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
        int x2 = u.x > x ? x + 1 : u.x < x ? x - 1 : x;
        int y = (int) u.y;
        int y2 = u.y > y ? y + 1 : u.y < y ? y - 1 : y;

        if (oob(x, y) || oob(x2, y) || oob(x, y2) || oob(x2, y2))
          return alpha;
      }
      return 1.0;
    }

    float traceDist(Vector2 p, Vector2 d) {
      // modified Bresenham
      Vector2 p2 = p + radius() * d;
      float alpha = traceLine(p, p2);
      return alpha * p.point(p2).length();
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
    static constexpr double maxDist = 8.0;

    Map *map;

    Vector2 p;  // position
    int action;  // direction

    void init(Map &map_) {
      map = &map_;
      reset();
    }

    std::optional<Vector2> pull = std::nullopt;

    void reset(std::optional<Vector2> loc = std::nullopt) {
      action = rand() % directions.size();
      p = loc.value_or(map->origin);
    }

    Vector2 d() {
      return directions[action];
    }

    void step() {
      p = p + d();
      map->visit(p.x, p.y);
    }

    double simpleReward() {
      //use with test map
      Vector2 d = map->target - p;
      Vector2 dnorm = d.normalized();
      Vector2 v = directions[action];

      double proxReward = 1.0 - (d.dot(d)) / double(map->w * map->w + map->h * map->h);
      CLAMP(proxReward, 0.0, 1.0);

      double facingReward = dnorm.dot(v);
      CLAMP(facingReward, 0.0, 1.0);

      // Final reward
      return 0.9 * proxReward + 0.1 * facingReward;
    }

    double avoidReward() {
      Vector2 d = p.point(map->target);
      Vector2 dnorm = d.normalized();
      Vector2 v = directions[action];

      double reward = 1.0 - (d.dot(d)) / map->radiusSq();
      CLAMP(reward, 0.0, 1.0);

      double facingReward = dnorm.dot(v);
      CLAMP(facingReward, 0.0, 1.0);

      double hitPenalty = traceHit(v);
      // return 0.3 * reward + facingReward - 1.3 * hitPenalty;
      return reward + 0.3 * facingReward - 0.3 * hitPenalty;
    }

    float traceLine(std::optional<Vector2> direction = std::nullopt) {
      Vector2 p2 = p + maxDist * direction.value_or(d());
      return map->traceLine(p, p2);
    }

    bool traceHit(std::optional<Vector2> direction = std::nullopt) {
      Vector2 p2 = p + maxDist * direction.value_or(d());
      return map->traceLine(p, p2) < 1.0;
    }

    int nnSetup(double *inputValues) {
      // normalize inputs
      double temp[NN_MAX_NEURONS];
      double *inputs = !inputValues ? temp : inputValues;

      int k = 0;
      inputs[k++] = p.x * map->downscale;
      inputs[k++] = p.y * map->downscale;
      inputs[k++] = map->target.x * map->downscale;
      inputs[k++] = map->target.y * map->downscale;
      //inputs[k++] = traceHit();
      for (auto d : directions) {
        inputs[k++] = traceHit(d);
      }
      return k;
    }

    Pixel24 color() {
      return Pixel24(255, 0, 255);
    }

    Pixel24 blockerColor() {
      return Pixel24(255, 255, 0);
    }

  };

  double learn = 0.1;  // neural network learning rate
  double lambda = 0.0003;  // l2 decay
  double gamma = 0.7;   // q learning discount factor
  double epsilon = 0.2;  // epsilon greedy epsilon value
  double alpha = 0.3;  // Bellman learning rate

  Map map;
  Agent explorer;

  struct RL {
    Map &map;
    Agent &explorer;
    RL_agent_t ai;
    RL(Map &map_, Agent &agent_)
        :
        map(map_),
        explorer(agent_),
        ai(nullptr) {
    }
    static double reward(RL_agent_state_t state) {
      RL *rl = (RL*) (state);
      //return rl->explorer.simpleReward();
      return rl->explorer.avoidReward();

    }
    static void set(RL_agent_state_t state, double *inputValues) {
      RL *rl = (RL*) (state);
      rl->explorer.nnSetup(inputValues);

    }

    static void act(RL_agent_state_t state, int action) {
      RL *rl = (RL*) (state);
      rl->explorer.action = action;
      if (!rl->explorer.traceHit())
        rl->explorer.step();
    }
  };
  RL *rl = nullptr;
  void init() {

    map.init("map1.bmp");
    explorer.init(map);

    NN_info_t info;
    info.activation = NN_sigmoid;
    info.learning_rate = learn;
    info.l2_decay = lambda;
    info.hidden_layers_size = 1;
    info.neurons_per[0] = 100;
    info.input_size = explorer.nnSetup(nullptr);
    info.output_size = directions.size();
    printf("%s:input size: %d\n", __FUNCTION__, info.input_size);
    printf("%s:output size: %d\n", __FUNCTION__, info.output_size);

    rl = new RL(map, explorer);
    rl->ai = RL_init(RL_sarsa, alpha, epsilon, gamma, &info, RL::set, RL::reward, RL::act, rl);
  }

  void explore() {
    RL_step(rl->ai);
  }

  ~Game() {
    RL_export_neural_network(rl->ai, "nn.txt");
    RL_term(&rl->ai);
    delete rl;
  }

};
