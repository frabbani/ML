#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "neural.h"

#define RNN_MAX_DEPTH 40

typedef enum {
  RNN_seq_to_one,
  RNN_seq_to_seq
} RNN_mode_t;

typedef struct {
  RNN_mode_t mode;
  double learning_rate;
  double beta;
  int input_size;
  int output_size;
  int hidden_layers_size;
  int bptt_depth;
  int neurons_per[NN_MAX_HIDDEN_LAYERS];
} RNN_info_t;

typedef struct {
  double weights[NN_MAX_NEURONS];
  double recurrent_weights[NN_MAX_NEURONS];
  double bias;
  struct{
    double weights[NN_MAX_NEURONS];
    double recurrent_weights[NN_MAX_NEURONS];
    double bias;
  }moment;

  double history[RNN_MAX_DEPTH];
  double delta[RNN_MAX_DEPTH];
} RNN_neuron_t;

typedef struct {
  double values[RNN_MAX_DEPTH][NN_MAX_NEURONS];
} RNN_sequence_t;

typedef struct RNN_neural_layer_s {
  int size;
  NN_layer_type_t type;
  RNN_neuron_t neurons[NN_MAX_NEURONS];
  union {
    struct RNN_neural_layer_s *feed;
    const RNN_sequence_t *input;
  };
} RNN_neural_layer_t;

typedef struct {
  RNN_info_t info;
  RNN_sequence_t input;
  RNN_neural_layer_t hidden_layers[NN_MAX_HIDDEN_LAYERS];
  RNN_neural_layer_t output_layer;
  RNN_sequence_t target;
  double prediction[NN_MAX_NEURONS];  //latest predictino
  int t;
  double beta_decay;
} RNN_neural_network_t;

typedef struct {
  int grad_count;
  int recur_grad_count;
  int delta_count;
  double grad_min;
  double grad_max;
  double grad_mean;
  double recur_grad_min;
  double recur_grad_max;
  double recur_grad_mean;
  double delta_min;
  double delta_max;
  double delta_mean;

} RNN_metrics_t;

void RNN_init_neural_network(RNN_neural_network_t *rnn, const RNN_info_t *params);
double RNN_forward_propagate(RNN_neural_network_t *rnn, const double *input, const double *target);
void RNN_backward_propagate(RNN_neural_network_t *rnn, RNN_metrics_t *metrics);
double RNN_train_neural_network(RNN_neural_network_t *rnn, const double *input, const double *target);
void RNN_reset_history(RNN_neural_network_t *rnn);

#ifdef __cplusplus
}
#endif
