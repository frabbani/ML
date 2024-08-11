#pragma once

#include "vector.h"
#include "neural.h"

struct Game {
  Vector2 runner;
  Vector2 chaser;

  std::vector<Vector2> trainingPts;
  std::vector<Vector2> trainingPts2;

  NN_neural_network_t *nn = nullptr;
  double learn = 0.003;

  void run(int xDir, int yDir, float speed) {
    runner = runner + speed * Vector2(xDir, yDir);
  }

  Game() {
    nn = new NN_neural_network_t;

    NN_info_t info;
    info.activation = NN_tanh;
    info.hidden_layers_size = 1;
    info.input_size = 4;
    info.output_size = 2;
    for (int i = 0; i < info.hidden_layers_size; i++)
      info.neurons_per[i] = 6;
    NN_init_neural_network(nn, &info);

    for (int i = 0; i < 512; i++) {
      Vector2 v;
      v.x = NN_random(1, -0.5);
      v.y = NN_random(1, -0.5);
      trainingPts.push_back(v);
      v.x = NN_random(1, -0.5);
      v.y = NN_random(1, -0.5);
      trainingPts2.push_back(v);
    }
  }

  Vector2 targetSetup() {
    Vector2 d;
    if (runner.x < chaser.x)
      d.x = -1.0;
    if (runner.x > chaser.x)
      d.x = +1.0;
    if (runner.y < chaser.y)
      d.y = -1.0;
    if (runner.y > chaser.y)
      d.y = +1.0;
    return d;
  }

  void trainChaser() {
    auto setup_target = [&]() {
      Vector2 d;
      Vector2 c(nn->input[0], nn->input[1]);
      Vector2 p(nn->input[2], nn->input[3]);
      if (p.x < c.x)
        d.x = -1.0;
      if (p.x > c.x)
        d.x = +1.0;
      if (p.y < c.y)
        d.y = -1.0;
      if (p.y > c.y)
        d.y = +1.0;
      return d;
    };

    size_t epochs = 0;
    for (auto p : trainingPts) {
      for (auto p2 : trainingPts2) {
        nn->input[0] = p.x;
        nn->input[1] = p.y;
        nn->input[2] = p2.x;
        nn->input[3] = p2.y;

        auto d = setup_target();
        nn->target[0] = d.x;
        nn->target[1] = d.y;

        NN_forward_propagate(nn);
        NN_backward_propagate(nn, learn);
        epochs++;
      }
    }
    printf("# of epochs: %zu\n", epochs);
  }

  void chase() {
    nn->input[0] = chaser.x;
    nn->input[1] = chaser.y;
    nn->input[2] = runner.x;
    nn->input[3] = runner.y;

    NN_forward_propagate(nn);

    float x = nn->prediction[0];  // < -0.5f ? -1.0f : nn->prediction[0] > 0.5f ?  +1.0f : 0.0f;
    float y = nn->prediction[1];  // < -0.5f ? -1.0f : nn->prediction[1] > 0.5f ?  +1.0f : 0.0f;

    chaser = chaser + Vector2(x, y) * 0.005;

//    printf("step %d\n", steps);
//    printf(" * input...: %lf, %lf\n", nn->prediction[0], nn->prediction[1]);
//    printf(" * position: %lf, %lf\n", chaser.x, chaser.y);
  }

  ~Game() {
    delete nn;
  }
}
;
