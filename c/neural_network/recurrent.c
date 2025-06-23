#include "recurrent.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP( v, l, h ){ v = v < (l) ? (l) : v > (h) ? (h) : v; }

extern double sigmoid_act(double x);
extern double sigmoid_deriv(double x);
extern double tanh_act(double x);
extern double tanh_deriv(double x);
extern double relu_act(double x);
extern double relu_deriv(double x);
extern double leaky_relu_act(double x, double alpha);
extern double leaky_relu_deriv(double x, double alpha);

double output_act(double x) {
  return x;
  //return sigmoid_act(x);
}

double output_deriv(double x) {
  return 1.0;
  //return sigmoid_deriv(x);
}

double hidden_act(double x) {
  return tanh_act(x);
  //return leaky_relu_act(x, 0.1);
}

double hidden_deriv(double x) {
  return tanh_deriv(x);
  //return leaky_relu_deriv(x, 0.1);
}

static void init_recurrent_neuron(RNN_neuron_t *neuron, int m, int n, int d) {
  neuron->bias = 0.0;

  double lim = sqrt(6.0 / (double) (m + n));  // Xaviar/Glorot
  for (int i = 0; i < m; i++)
    neuron->weights[i] = NN_random(2.0 * lim, -lim);

  lim = sqrt(1.0 / n);  // He
  for (int i = 0; i < n; i++)
    neuron->recurrent_weights[i] = NN_random(2.0 * lim, -lim);

  for (int i = 0; i < d; i++)
    neuron->history[i] = neuron->delta[i] = 0.0;
}

static void init_recurrent_neural_first_hidden_layer(RNN_neural_layer_t *layer, int size, int input_size, const RNN_sequence_t *input, int depth) {
  layer->type = NN_first;
  layer->size = size;
  CLAMP(layer->size, 1, NN_MAX_NEURONS);
  layer->input = input;
  for (int i = 0; i < layer->size; i++)
    init_recurrent_neuron(&layer->neurons[i], input_size, layer->size, depth);
}

static void init_recurrent_neural_hidden_layer(RNN_neural_layer_t *layer, RNN_neural_layer_t *previous_layer, int size, int depth) {
  layer->type = NN_hidden;
  layer->size = size;
  CLAMP(layer->size, 1, NN_MAX_NEURONS);
  layer->feed = previous_layer;
  for (int i = 0; i < layer->size; i++)
    init_recurrent_neuron(&layer->neurons[i], layer->feed->size, layer->size, depth);
}

static void init_recurrent_neural_output_layer(RNN_neural_layer_t *layer, RNN_neural_layer_t *previous_layer, int size, int depth) {
  layer->type = NN_output;
  layer->size = size;
  CLAMP(layer->size, 1, NN_MAX_NEURONS);
  layer->feed = previous_layer;
  for (int i = 0; i < layer->size; i++)
    init_recurrent_neuron(&layer->neurons[i], layer->feed->size, layer->size, depth);
}

static void recurrent_neural_layer_propagate_hidden(RNN_neural_layer_t *layer, int input_size, int now) {
  int then = (now - 1 + RNN_MAX_DEPTH) % RNN_MAX_DEPTH;
  for (int i = 0; i < layer->size; i++) {
    RNN_neuron_t *neuron = &layer->neurons[i];
    double sum = neuron->bias;

    // input sum
    if (layer->type == NN_first) {
      for (int j = 0; j < input_size; j++)
        sum += neuron->weights[j] * layer->input->values[now][j];
    } else {
      for (int j = 0; j < layer->feed->size; j++)
        sum += neuron->weights[j] * layer->feed->neurons[j].history[now];
    }

    // recurrent sum
    for (int j = 0; j < layer->size; j++)
      sum += neuron->recurrent_weights[j] * layer->neurons[j].history[then];

    neuron->history[now] = hidden_act(sum);
  }
}

static void recurrent_neural_layer_propagate_output(RNN_neural_layer_t *layer, int now) {
  if (layer->type != NN_output)
    return;

  for (int i = 0; i < layer->size; i++) {
    RNN_neuron_t *neuron = &layer->neurons[i];
    double sum = neuron->bias;
    for (int j = 0; j < layer->feed->size; j++)
      sum += neuron->weights[j] * layer->feed->neurons[j].history[now];
    neuron->history[now] = output_act(sum);
  }
}

void RNN_init_neural_network(RNN_neural_network_t *rnn, const RNN_info_t *params) {
  rnn->info.hidden_layers_size = params->hidden_layers_size;
  CLAMP(rnn->info.hidden_layers_size, 1, NN_MAX_HIDDEN_LAYERS);
  rnn->info.input_size = params->input_size;
  CLAMP(rnn->info.input_size, 1, NN_MAX_NEURONS);
  rnn->info.output_size = params->output_size;
  CLAMP(rnn->info.output_size, 1, NN_MAX_NEURONS);
  for (int i = 0; i < rnn->info.hidden_layers_size; i++) {
    rnn->info.neurons_per[i] = params->neurons_per[i];
    CLAMP(rnn->info.neurons_per[i], 1, NN_MAX_NEURONS);
  }
  rnn->info.learning_rate = fabs(params->learning_rate);
  rnn->info.bptt_depth = params->bptt_depth;
  CLAMP(rnn->info.bptt_depth, 1, RNN_MAX_DEPTH-1);  //allow for oldest - 1

  for (int i = 0; i < rnn->info.bptt_depth; i++) {
    for (int j = 0; j < rnn->info.input_size; j++)
      rnn->input.values[i][j] = 0.0;
    for (int j = 0; j < rnn->info.output_size; j++)
      rnn->target.values[i][j] = 0.0;
  }

  init_recurrent_neural_first_hidden_layer(&rnn->hidden_layers[0], rnn->info.neurons_per[0], rnn->info.input_size, &rnn->input, rnn->info.bptt_depth);
  int nls = rnn->info.hidden_layers_size;
  for (int i = 1; i < nls; i++)
    init_recurrent_neural_hidden_layer(&rnn->hidden_layers[i], &rnn->hidden_layers[i - 1], rnn->info.neurons_per[i], rnn->info.bptt_depth);
  init_recurrent_neural_output_layer(&rnn->output_layer, &rnn->hidden_layers[nls - 1], rnn->info.output_size, rnn->info.bptt_depth);
  rnn->t = 0;
}

double RNN_forward_propagate(RNN_neural_network_t *rnn, const double *input, const double *target) {
  rnn->t++;
  int now = rnn->t % RNN_MAX_DEPTH;

  for (int i = 0; i < rnn->info.input_size; i++)
    rnn->input.values[now][i] = input[i];

  for (int i = 0; i < rnn->info.output_size; i++)
    rnn->target.values[now][i] = target[i];

  for (int i = 0; i < rnn->info.hidden_layers_size; i++) {
    recurrent_neural_layer_propagate_hidden(&rnn->hidden_layers[i], rnn->info.input_size, now);
  }
  recurrent_neural_layer_propagate_output(&rnn->output_layer, now);

  double mse = 0.0;
  for (int i = 0; i < rnn->info.output_size; i++) {
    rnn->prediction[i] = rnn->output_layer.neurons[i].history[now];
    double diff = rnn->prediction[i] - rnn->target.values[now][i];
    mse += (diff * diff);
  }
  return mse / (double) rnn->info.output_size;
}


void RNN_backward_propagate(RNN_neural_network_t *rnn, RNN_metrics_t *metrics) {
  double learning_rate = rnn->info.learning_rate / (double) rnn->info.bptt_depth;
  int depth = rnn->info.bptt_depth;
  int output_size = rnn->info.output_size;
  RNN_neural_layer_t *output_layer = &rnn->output_layer;
  RNN_neuron_t *output_neurons = output_layer->neurons;

  if(metrics){
    metrics->grad_count = metrics->recur_grad_count = metrics->delta_count = 0;
    metrics->grad_min = metrics->recur_grad_min = metrics->delta_min = +INFINITY;
    metrics->grad_max = metrics->recur_grad_max = metrics->delta_max = -INFINITY;
    metrics->grad_mean = metrics->recur_grad_mean = metrics->delta_mean = 0.0;
  }

  for (int d = 0; d < depth; d++) {
    int when = (rnn->t - d + RNN_MAX_DEPTH) % RNN_MAX_DEPTH;
    for (int i = 0; i < output_size; i++) {
      double output = output_neurons[i].history[when];
      output_neurons[i].delta[when] = (output - rnn->target.values[when][i]) * output_deriv(output);
      if(metrics){
        metrics->delta_count++;
        metrics->delta_min = MIN(output_neurons->delta[when], metrics->delta_min);
        metrics->delta_max = MAX(output_neurons->delta[when], metrics->delta_max);
        metrics->delta_mean += output_neurons->delta[when];
      }
    }
  }

  for (int d = 0; d < depth; d++) {
    int now = (rnn->t - d + RNN_MAX_DEPTH) % RNN_MAX_DEPTH;
    int then = (rnn->t - d - 1 + RNN_MAX_DEPTH) % RNN_MAX_DEPTH;
    for (int l = rnn->info.hidden_layers_size - 1; l >= 0; l--) {
      RNN_neural_layer_t *layer = &rnn->hidden_layers[l];
      RNN_neural_layer_t *next_layer = &rnn->output_layer;
      if (l < rnn->info.hidden_layers_size - 1)
        next_layer = &rnn->hidden_layers[l + 1];

      for (int i = 0; i < layer->size; i++) {
        RNN_neuron_t *neuron = &layer->neurons[i];
        double sum = 0.0;
        for (int j = 0; j < layer->size; j++)
          sum += layer->neurons[j].delta[then] * neuron->recurrent_weights[j];

        for (int j = 0; j < next_layer->size; j++)
          sum += next_layer->neurons[j].delta[now] * next_layer->neurons[j].weights[i];
        neuron->delta[now] = sum * hidden_deriv(neuron->history[now]);
        if(metrics){
          metrics->delta_count++;
          metrics->delta_min = MIN(neuron->delta[now], metrics->delta_min);
          metrics->delta_max = MAX(neuron->delta[now], metrics->delta_max);
          metrics->delta_mean += neuron->delta[now];
        }
      }
    }
  }

  for (int d = 0; d < depth; d++) {
    int now = (rnn->t - d + RNN_MAX_DEPTH) % RNN_MAX_DEPTH;
    int then = (rnn->t - d - 1 + RNN_MAX_DEPTH) % RNN_MAX_DEPTH;

    for (int i = 0; i < output_layer->size; i++) {
      RNN_neuron_t *neuron = &output_layer->neurons[i];
      double delta = neuron->delta[now];
      neuron->bias -= learning_rate * delta;
      for (int j = 0; j < output_layer->feed->size; j++) {
        double grad = delta * output_layer->feed->neurons[j].history[now];
        if(metrics){
          metrics->grad_count++;
          metrics->grad_min = MIN(grad, metrics->grad_min);
          metrics->grad_max = MAX(grad, metrics->grad_max);
          metrics->grad_mean += grad;
        }
        neuron->weights[j] -= learning_rate * grad;
      }
      for (int j = 0; j < output_layer->size; j++)
        neuron->recurrent_weights[j] = 0.0;
    }

    for (int l = 0; l < rnn->info.hidden_layers_size; l++) {
      RNN_neural_layer_t *layer = &rnn->hidden_layers[l];

      for (int i = 0; i < layer->size; i++) {
        RNN_neuron_t *neuron = &layer->neurons[i];
        double delta = neuron->delta[now];
        neuron->bias -= learning_rate * delta;

        for (int j = 0; j < layer->size; j++) {
          double grad = delta * layer->neurons[j].history[then];
          if(metrics){
            metrics->recur_grad_count++;
            metrics->recur_grad_min = MIN(grad, metrics->recur_grad_min);
            metrics->recur_grad_max = MAX(grad, metrics->recur_grad_max);
            metrics->recur_grad_mean += grad;
          }
          neuron->recurrent_weights[j] -= learning_rate * grad;
        }

        if (layer->type == NN_first) {
          double *input = rnn->input.values[now];
          for (int j = 0; j < rnn->info.input_size; j++) {
            double grad = delta * input[j];
            if(metrics){
              metrics->grad_count++;
              metrics->grad_min = MIN(grad, metrics->grad_min);
              metrics->grad_max = MAX(grad, metrics->grad_max);
              metrics->grad_mean += grad;
            }
            neuron->weights[j] -= learning_rate * grad;
          }

        } else {
          RNN_neural_layer_t *prev_layer = layer->feed;
          for (int j = 0; j < prev_layer->size; j++) {
            double grad = delta * prev_layer->neurons[j].history[now];
            if(metrics){
              metrics->grad_count++;
              metrics->grad_min = MIN(grad, metrics->grad_min);
              metrics->grad_max = MAX(grad, metrics->grad_max);
              metrics->grad_mean += grad;
            }
            neuron->weights[j] -= learning_rate * grad;
          }
        }
      }
    }
  }
  if(metrics){
    if(metrics->grad_count)
      metrics->grad_mean /= (double)metrics->grad_count;
    if(metrics->recur_grad_count)
      metrics->recur_grad_mean /= (double)metrics->recur_grad_count;
    if(metrics->delta_count)
      metrics->delta_mean /= (double)metrics->delta_count;
  }
}

double RNN_train_neural_network(RNN_neural_network_t *rnn, const double *input, const double *target) {
  RNN_forward_propagate(rnn, input, target);
  if (0 == (rnn->t % rnn->info.bptt_depth))
    RNN_backward_propagate(rnn, NULL);  //not collecting metrics for now

  int now = rnn->t % RNN_MAX_DEPTH;
  double mse = 0.0;
  for (int i = 0; i < rnn->info.output_size; i++) {
    rnn->prediction[i] = rnn->output_layer.neurons[i].history[now];
    double diff = rnn->prediction[i] - rnn->target.values[now][i];
    mse += diff * diff;
  }

  return mse / (double) rnn->info.output_size;


}

void RNN_reset_history(RNN_neural_network_t *rnn) {
  for (int l = 0; l < rnn->info.hidden_layers_size; l++) {
    RNN_neural_layer_t *layer = &rnn->hidden_layers[l];
    for (int i = 0; i < layer->size; i++) {
      RNN_neuron_t *neuron = &layer->neurons[i];
      for (int j = 0; j < rnn->info.bptt_depth; j++) {
        neuron->delta[j] = neuron->history[j] = 0.0;
      }
    }
  }
}
