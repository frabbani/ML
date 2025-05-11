#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "reinforce.h"

static RL_type_t rl_type = RL_sarsa;
static RL_bool rl_inited = RL_false;
static NN_neural_network_t *rl_nn = RL_nullptr;
static double *rl_qs[2] = { RL_nullptr, RL_nullptr };
static int rl_qcount = 0;
static RL_act_cb rl_act = RL_nullptr;
static RL_set_inputs_cb rl_set = RL_nullptr;
static RL_reward_cb rl_reward = RL_nullptr;
static void *rl_param = RL_nullptr;

static double rl_alpha = 0.0;
static double rl_epsilon = 0.0;
static double rl_gamma = 0.0;
static RL_action_t rl_action = { 0, RL_false };

#define CURR_QS  0
#define NEXT_QS  1

void RL_init(RL_type_t type, double alpha, double epsilon, double gamma,
             const NN_info_t *nn_info, RL_set_inputs_cb set_inputs,
             RL_reward_cb reward, RL_act_cb act, void *param) {
  if (!rl_nn)
    rl_nn = malloc(sizeof(NN_neural_network_t));
  NN_info_t info;
  memcpy(&info, nn_info, sizeof(NN_info_t));
  info.input_size += 2;  // action
  NN_init_neural_network(rl_nn, &info);

  rl_type = type;
  rl_alpha = alpha;
  rl_epsilon = epsilon;
  rl_gamma = gamma;

  rl_qcount = rl_nn->output_size;
  rl_qs[0] = malloc(sizeof(double) * rl_qcount);
  rl_qs[1] = malloc(sizeof(double) * rl_qcount);

  rl_act = act;
  rl_set = set_inputs;
  rl_reward = reward;
  rl_param = param;

  rl_nn->input[0] = (double) rl_action.exploratory;
  rl_nn->input[1] = (double) rl_action.taken;
  rl_set(rl_param, &rl_nn->input[2]);

  rl_inited = RL_true;
}

static int q_max(int type) {
  double *qs = type == CURR_QS ? rl_qs[CURR_QS] : rl_qs[NEXT_QS];
  int best = 0;
  double q = qs[0];
  for (int i = 1; i < rl_qcount; i++) {
    if (qs[i] > q) {
      best = i;
      q = qs[i];
    }
  }
  return best;
}

static RL_action_t e_greedy() {
  RL_action_t action = { 0, RL_false };
  if (NN_random(1.0, 0.0) < rl_epsilon) {
    action.exploratory = RL_true;
    action.taken = (int) NN_random((double) rl_qcount, 0.0);
  } else {
    action.taken = q_max(CURR_QS);
    action.exploratory = RL_false;
  }
  return action;

}

void RL_step() {
  if (!rl_inited)
    return;

  rl_nn->input[0] = (double) rl_action.exploratory;
  rl_nn->input[1] = (double) rl_action.taken;
  rl_set(rl_param, &rl_nn->input[2]);

  NN_forward_propagate(rl_nn);
  for (int i = 0; i < rl_qcount; i++) {
    rl_qs[CURR_QS][i] = rl_nn->output_layer.neurons[i].value;
  }
  RL_action_t last_action = rl_action;
  rl_action = e_greedy();

  rl_act(rl_param, rl_action.taken);
  double reward = rl_reward(rl_param);

  rl_nn->input[0] = (double) rl_action.exploratory;
  rl_nn->input[1] = (double) rl_action.taken;
  rl_set(rl_param, &rl_nn->input[2]);

  NN_forward_propagate(rl_nn);
  for (int i = 0; i < rl_qcount; i++) {
    rl_qs[NEXT_QS][i] = rl_nn->output_layer.neurons[i].value;
  }

  double target = 0.0;
  for (int i = 0; i < rl_qcount; i++)
    rl_nn->target[i] = rl_qs[CURR_QS][i];

  if (RL_sarsa == rl_type) {
    target = reward + rl_gamma * rl_qs[NEXT_QS][rl_action.taken];  // SARSA
    rl_nn->target[rl_action.taken] += rl_alpha
        * (target - rl_qs[CURR_QS][rl_action.taken]);
  }
  if (RL_qlearn == rl_type) {
    target = reward + rl_gamma * q_max(NEXT_QS);  //Q-Learning
    rl_nn->target[last_action.taken] += rl_alpha
        * (target - rl_qs[CURR_QS][last_action.taken]);

  }
  NN_backward_propagate(rl_nn);
}

void RL_term() {
  if (rl_nn) {
    free(rl_nn);
    rl_nn = RL_nullptr;
  }
  if (rl_qs[0])
    free(rl_qs[0]);
  if (rl_qs[1])
    free(rl_qs[1]);
  rl_qs[0] = rl_qs[1] = RL_nullptr;

  rl_qcount = 0;
  rl_act = RL_nullptr;
  rl_set = RL_nullptr;
  rl_reward = RL_nullptr;
  rl_param = RL_nullptr;
  rl_inited = RL_false;
}

double RL_Qvalue(int index) {
  if (rl_inited) {
    index = index < 0 ? 0 : index >= rl_qcount ? rl_qcount - 1 : index;
    return rl_qs[CURR_QS][index];
  }
  return -1.0;
}

void RL_export_neural_network(const char *filename) {
  NN_export_neural_network(rl_nn, filename);
}
