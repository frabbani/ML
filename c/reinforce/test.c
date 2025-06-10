#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reinforce.h"

// one-hot vector of size 5

typedef struct {
  int position;  // current position on a 1D grid [0..4]
  int steps;     // step counter for episode
} grid_agent_state_t;

void set_input_cb(RL_agent_state_t state, double *input) {
  grid_agent_state_t *s = (grid_agent_state_t*) state;
  for (int i = 0; i < 5; ++i)
    input[i] = (i == s->position) ? 1.0 : 0.0;
}

void act_cb(RL_agent_state_t state, int action) {
  grid_agent_state_t *s = (grid_agent_state_t*) state;
  if (action == 0 && s->position > 0)
    s->position--;
  else if (action == 1 && s->position < 4)
    s->position++;
  s->steps++;
}

double reward_cb(RL_agent_state_t state) {
  grid_agent_state_t *s = (grid_agent_state_t*) state;
  return (s->position == 4) ? 1.0 : 0.0;
}

int main() {
  grid_agent_state_t agent_state = { .position = 0, .steps = 0 };

  NN_info_t nn_info;
  nn_info.activation = NN_relu;
  nn_info.learning_rate = 0.01;
  nn_info.l2_decay = 0.0003;
  nn_info.input_size = 5;
  nn_info.output_size = 2;  // left/right is 0/1
  nn_info.hidden_layers_size = 1;
  nn_info.neurons_per[0] = 8;
  NN_seed_random(42);  // Optional: set RNG seed for reproducibility

  RL_agent_t agent = RL_init(RL_qlearn,
      0.1,             // alpha (learning rate)
      0.2,             // epsilon (exploration)
      0.99,            // gamma (discount factor)
      &nn_info, set_input_cb, reward_cb, act_cb, &agent_state);

  for (int ep = 0; ep < 100; ep++) {
    agent_state.position = 0;
    agent_state.steps = 0;
    for (int i = 0; i < 20; i++) {
      RL_step(agent);
      if (agent_state.position == 4)
        break;
    }

    printf("Episode %3d: reached %d in %2d steps? %s\n", ep,
           agent_state.position, agent_state.steps,
           (agent_state.position == 4) ? "yes!" : "no");
  }

  RL_export_neural_network(agent, "nn.txt");
  RL_term(&agent);
  return 0;
}
