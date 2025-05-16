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
    static constexpr double ewma = 0.03;
    static constexpr double ewma2 = 0.005;
    static constexpr double maxDist = 8.0;
    static constexpr double lingerDist = 5.0;
    static constexpr double linger2Dist = 9.0;
    static constexpr double markDist = 16.0;

    Vector2 p;              // position
    Vector2 c;              // short term history
    Vector2 c2;             // long term history
    std::array<Vector2, 8> blockers;
    double stall = 0.0;
    int steps = 0;
    Map *map;
    void init(Map &map_) {
      this->map = &map_;
      reset();
    }

    int actionTaken;

    std::optional<Vector2> pull = std::nullopt;

    void reset(std::optional<Vector2> loc = std::nullopt) {
      if (map) {
        actionTaken = rand() % directions.size();
      }
      p = c = c2 = loc.value_or(map->origin);
      for (auto &blocker : blockers)
        blocker = map->origin;
      stall = 0.0;
      steps = 0;
    }

    Vector2 d() {
      return directions[actionTaken];
    }

    void step() {
      p = p + d();
      map->visit(p.x, p.y);
      auto d = p - c;
      if (d.dot(d) >= (lingerDist * lingerDist))
        c = LERP(c, p, ewma);
      d = p - c2;
      if (d.dot(d) >= (linger2Dist * linger2Dist))
        c2 = LERP(c2, p, ewma2);
      stall += 0.001;
      steps++;
    }

    bool lineOfSight(Vector2 target) {
      return map->traceLine(p, target) < 1.0 ? false : true;
    }

    void addBlocker(Vector2 v) {
      //Vector2 e = p + maxDist * v;
      //float alpha = map->traceLine(p, e);
      //v = p + alpha * (p.point(e));
      static int c = 0;
      int n = int(blockers.size());
      c++;
      c %= n;
      blockers[c] = v;
    }

    void maybeAddBlocker() {
      static int counter = 0;
      Vector2 u = p.point(c);
      if (u.dot(u) > markDist * markDist)
        counter = 0;
      else {
        counter++;
        if (counter > 50) {
          addBlocker(c);
          counter = -50 * 10;
        }
      }
    }
    double reward() {
      double exitReward = 0.0;
      double lingerPenalty = 0.0;
      double hitPenalty = 0.0;
      double projection = 0.0;
      double repelPenalty = 0.0;
      double pullReward = 0.0;

      maybeAddBlocker();

      Vector2 dir;
      Vector2 dirnorm;
      double lenSq = 0.0;
      Vector2 act = directions[actionTaken];

      dir = p.point(map->target);
      dirnorm = dir.normalized();
      exitReward = 1.0
          - (dir.dot(act)) / double(map->w * map->w + map->h * map->h);
      CLAMP(exitReward, 0.0, 1.0);
      double facingReward = dirnorm.dot(act);
      CLAMP(facingReward, 0.0, 1.0);
      hitPenalty = traceHit();
      exitReward = 0.3 * exitReward + facingReward - 1.3 * hitPenalty;

      /*
       exitReward = 1.0;
       projection = (map->target - p).normalized().dot(act);
       CLAMP(projection, -0.1, 1.0);
       exitReward *= projection;
       */

      //if (!lineOfSight(map->target))
      //  exitReward *= 0.6f;
//      for (auto r : repellants) {
//        Vector2 to = r - p;
//        lenSq = to.dot(to);
//        if (lenSq > lingerDist * lingerDist) {
//          double len = sqrt(lenSq);
//          projection = to.normalized().dot(act);
//          CLAMP(projection, 0.0, 1.0);
//          float factor = (1.0 - len * map->downscale);
//          CLAMP(factor, 0.0, 1.0);
//          factor *= lineOfSight(r) ? 1.0 : 0.0;
//          repelPenalty += factor * projection;
//        } else
//          repelPenalty += 1.0;
//      }
      if (pull.has_value()) {
        Vector2 to = *pull - p;
        lenSq = to.dot(to);
        if (lenSq > 0.0 && lenSq > lingerDist * lingerDist) {
          projection = to.normalized().dot(act);
          CLAMP(projection, 0.0, 1.0);
          pullReward = projection;
        }
      }

      dir = c - p;  // point towards center
      lenSq = dir.dot(dir);
      if (lenSq > 0.0 && lenSq <= lingerDist * lingerDist) {
        lingerPenalty = 1.0;
        projection = dir.normalized().dot(act);
        CLAMP(projection, 0.0, 1.0);
        lingerPenalty *= projection;
      }

      hitPenalty = traceHit();

      double reward = exitReward + 2.0 * pullReward - 0.9 * hitPenalty
          - 0.1 * lingerPenalty - 0.7 * repelPenalty;

      CLAMP(reward, -1.0, 1.0);
      return reward;

    }

    double simpleReward() {
      //use with test map
      Vector2 d = map->target - p;
      Vector2 dnorm = d.normalized();
      Vector2 v = directions[actionTaken];

      double proxReward = 1.0
          - (d.dot(d)) / double(map->w * map->w + map->h * map->h);
      CLAMP(proxReward, 0.0, 1.0);

      double facingReward = dnorm.dot(v);
      CLAMP(facingReward, 0.0, 1.0);

      // Final reward
      return proxReward * 0.3 + facingReward;
    }

    double stagnate(double radius) {
      double len = (p - c).length();
      double len2 = (p - c2).length();

      return 1.0 - 0.5 * (len / radius + len2 / radius);
      // Both distances must be small to apply penalty
      if (len < lingerDist && len2 < linger2Dist)
        return 1.0
            - 0.5 * ((len / lingerDist) * 0.5 - (len2 / linger2Dist) * 0.5);

      return 0.0;
    }

    double proximity(Vector2 e, Vector2 v, float radiusSq) {
      Vector2 d = p.point(e);
      Vector2 dnorm = d.normalized();

      double reward = 1.0 - (d.dot(d) / radiusSq);
      CLAMP(reward, 0.0, 1.0);

      double facingReward = dnorm.dot(v);
      CLAMP(facingReward, 0.0, 1.0);

      double hitPenalty = map->traceLine(p, p + maxDist * v) < 1.0;
      //return 0.3 * reward + facingReward - 1.3 * hitPenalty;
      return reward + 0.3 * facingReward - 0.3 * hitPenalty;
    }

    double proximity2(Vector2 e, Vector2 v, float radiusSq) {
      if (map->traceLine(p, e) < 0.0)
        return 0.0;

      Vector2 d = p.point(e);
      Vector2 dnorm = d.normalized();
      if (d.dot(d) < lingerDist * lingerDist)
        return 1.0;

      double reward = 1.0 - (d.dot(d) / radiusSq);
      CLAMP(reward, 0.0, 1.0);

      double facingReward = dnorm.dot(v) > 0.0 ? 1.0 : 0.0;

      return 0.9 * reward + 0.1 * facingReward;
    }

    double backtrack(Vector2 v) {
      if ((p - c).length() < lingerDist && (p - c2).length() < linger2Dist) {
        Vector2 d = p.point(c2);
        double tug = v.dot(d.normalized());  // higher if facing away
        return tug;
      }
      return 0.0;
    }

    double avoidReward() {
      return proximity(map->target, directions[actionTaken], map->radiusSq());
    }

    double keepMovingReward() {
      return proximity(map->target, directions[actionTaken], map->radiusSq())
          - 0.2 * proximity(c, directions[actionTaken], map->radiusSq())
          - 0.2 * proximity(c2, directions[actionTaken], map->radiusSq())
          - stall;
    }

    double smartReward() {
      maybeAddBlocker();

      Vector2 v = directions[actionTaken];

      //if(traceHit(v) < 1.0)
      //  return 0.0;

      double blockerPenalty = 0.0;
      int n = int(blockers.size());
      for (int i = 0; i < n; i++) {
        Vector2 b = blockers[i];
        blockerPenalty += proximity(b, v, map->radiusSq());
      }

      return keepMovingReward() - 0.7 / double(n) * blockerPenalty;

      //double lingerPenalty = 0.0;
      //Vector2 d = p.point(c);  // point towards center
      //lingerPenalty += proximity(c, v, map->radiusSq());

//      if (lenSq > 0.0 && lenSq <= lingerDist * lingerDist) {
//        lingerPenalty = 1.0;
//        projection = dir.normalized().dot(act);
//        CLAMP(projection, 0.0, 1.0);
//        lingerPenalty *= projection;
//      }

      //return exitReward - 3.0 * blockerPenalty / double(n);  //- 0.3 * lingerPenalty;

      //if (!lineOfSight(map->target))
      //  exitReward *= 0.6f;

//      for (auto r : repellants) {
//        Vector2 to = r - p;
//        lenSq = to.dot(to);
//        if (lenSq > lingerDist * lingerDist) {
//          double len = sqrt(lenSq);
//          projection = to.normalized().dot(act);
//          CLAMP(projection, 0.0, 1.0);
//          float factor = (1.0 - len * map->downscale);
//          CLAMP(factor, 0.0, 1.0);
//          factor *= lineOfSight(r) ? 1.0 : 0.0;
//          repelPenalty += factor * projection;
//        } else
//          repelPenalty += 1.0;
//      }

//      if (pull.has_value()) {
//        Vector2 to = *pull - p;
//        lenSq = to.dot(to);
//        if (lenSq > 0.0 && lenSq > lingerDist * lingerDist) {
//          projection = to.normalized().dot(act);
//          CLAMP(projection, 0.0, 1.0);
//          pullReward = projection;
//        }
//      }
//

//      hitPenalty = traceHit();
//
//      double reward = exitReward + 2.0 * pullReward - 0.9 * hitPenalty
//          - 0.1 * lingerPenalty - 0.7 * repelPenalty;
//
//      CLAMP(reward, -1.0, 1.0);
//      return reward;

    }

    float traceLine(std::optional<Vector2> direction = std::nullopt) {
      Vector2 p2 = p + maxDist * direction.value_or(d());
      return map->traceLine(p, p2);
    }

    bool traceHit(std::optional<Vector2> direction = std::nullopt) {
      Vector2 p2 = p + maxDist * direction.value_or(d());
      return map->traceLine(p, p2) < 1.0;
    }

    float traceSight(std::optional<Vector2> direction = std::nullopt) {
      Vector2 p2 = p + map->radius() * direction.value_or(d());
      return map->traceLine(p, p2);
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
      inputs[k++] = traceHit();
      for (auto d : directions) {
        inputs[k++] = traceHit(d);
        //inputs[k++] = traceSight(d);
      }
      int n = int(blockers.size());
      for (int i = 0; i < n; i++) {
        inputs[k++] = blockers[i].x * map->downscale;
        inputs[k++] = blockers[i].y * map->downscale;
      }
      inputs[k++] = c.x * map->downscale;
      inputs[k++] = c.y * map->downscale;
      inputs[k++] = c2.x * map->downscale;
      inputs[k++] = c2.y * map->downscale;

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
      return rl->explorer.keepMovingReward();
    }
    static void set(RL_agent_state_t state, double *inputValues) {
      RL *rl = (RL*) (state);
      rl->explorer.nnSetup(inputValues);

    }

    static void act(RL_agent_state_t state, int action) {
      RL *rl = (RL*) (state);
      rl->explorer.actionTaken = action;
      if (!rl->explorer.traceHit())
        rl->explorer.step();
    }
  };
  RL *rl = nullptr;
  void init() {

    map.init("map1.bmp");
    explorer.init(map);

    NN_info_t info;
    info.activation = NN_relu;
    info.learning_rate = learn;
    info.l2_decay = lambda;
    info.hidden_layers_size = 2;
    info.neurons_per[0] = 90;
    info.neurons_per[1] = 90;
    info.input_size = explorer.nnSetup(nullptr);
    info.output_size = directions.size();
    printf("%s:input size: %d\n", __FUNCTION__, info.input_size);
    printf("%s:output size: %d\n", __FUNCTION__, info.output_size);

    rl = new RL(map, explorer);
    rl->ai = RL_init(RL_sarsa, alpha, epsilon, gamma, &info, RL::set,
                     RL::reward, RL::act, rl);
  }

  void explore() {
    RL_step(rl->ai);
    if(rl->explorer.steps > 400)
      rl->explorer.reset();
  }

  ~Game() {
    RL_export_neural_network(rl->ai, "nn.txt");
    RL_term(&rl->ai);
    delete rl;
  }

};
