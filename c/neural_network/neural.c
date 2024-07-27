#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "neural.h"

#define CLAMP( v, l, h ){ v = v < (l) ? (l) : v > (h) ? (h) : v; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

// might want to replace this - lol

double NN_random( double scale, double offset ) {
  union {
    uint32_t v;
    struct {
      uint8_t xyzw[4];
    };
  } r;
  r.xyzw[0] = rand() % 256;
  r.xyzw[1] = rand() % 256;
  r.xyzw[2] = rand() % 256;
  r.xyzw[3] = rand() % 256;
  double gen = (double) (r.v) / (double) (0xffffffff);
  return gen * scale + offset;
}


static double sigmoid_act(double x) {
  return 1.0 / (1.0 + exp(-x));
}

// Activation function derivatives
static double sigmoid_deriv(double x) {
  double s = sigmoid_act(x);
  return s * (1 - s);
}

static double tanh_act(double x) {
  return tanh(x);  // or use the formula directly
}

static double tanh_deriv(double x) {
  double tanh_x = tanh(x);
  return 1.0 - tanh_x * tanh_x;
}

static void init_neuron(NN_neuron_t *neuron, int n) {
  for (int i = 0; i < n; i++)
    neuron->weights[i] = NN_random( 2.0, -1.0 );
  neuron->bias = 0.0;
}

static void init_neural_layer(NN_neural_layer_t *layer, int size, NN_neural_layer_t *feed, int is_output) {
  layer->type = is_output ? NN_output : NN_hidden;
  layer->size = size <= 0 ? 1 : size > NN_MAX_NEURONS ? NN_MAX_NEURONS : size;
  layer->feed = feed;
  if (layer->feed) {
    for (int i = 0; i < layer->size; i++) {
      init_neuron(&layer->neurons[i], feed->size);
    }
  }
}

static void init_neural_first_hidden_layer(NN_neural_layer_t *layer, int size, int input_size, const double *input) {
  layer->type = NN_first;
  layer->size = size <= 0 ? 1 : size > NN_MAX_NEURONS ? NN_MAX_NEURONS : size;
  layer->input = input;
  for (int i = 0; i < layer->size; i++)
    init_neuron(&layer->neurons[i], input_size);
}

static void neural_layer_propagate(NN_neural_layer_t *layer, int input_size) {
  for (int i = 0; i < layer->size; i++) {
    NN_neuron_t *neuron = &layer->neurons[i];
    neuron->value = neuron->bias;
    if (layer->type == NN_first) {
      for (int j = 0; j < input_size; j++)
        neuron->value += neuron->weights[j] * layer->input[j];
    } else {
      for (int j = 0; j < layer->feed->size; j++)
        neuron->value += neuron->weights[j] * layer->feed->neurons[j].value;
    }
    neuron->value = tanh_act(neuron->value);
  }
}

static void neural_layer_propagate_regress(NN_neural_layer_t *layer) {
  for (int i = 0; i < layer->size; i++) {
    NN_neuron_t *neuron = &layer->neurons[i];
    neuron->value = neuron->bias;
    if (layer->type == NN_output) {
      for (int j = 0; j < layer->feed->size; j++)
        neuron->value += neuron->weights[j] * layer->feed->neurons[j].value;
    }
    //no activation!
  }
}

void NN_init_neural_network(NN_neural_network_t *nn, const NN_info_t *params) {
  nn->info.hidden_layers_size = params->hidden_layers_size;
  CLAMP(nn->info.hidden_layers_size, 1, NN_MAX_HIDDEN_LAYERS);
  nn->info.input_size = params->input_size;
  CLAMP(nn->info.input_size, 1, NN_MAX_NEURONS);
  nn->info.output_size = params->output_size;
  CLAMP(nn->info.output_size, 1, NN_MAX_NEURONS);
  for (int i = 0; i < nn->info.hidden_layers_size; i++) {
    nn->info.neurons_per[i] = params->neurons_per[i];
    CLAMP(nn->info.neurons_per[i], 1, NN_MAX_NEURONS);
  }

  init_neural_first_hidden_layer(&nn->hidden_layers[0], nn->info.neurons_per[0], nn->info.input_size, nn->input);

  int nls = nn->info.hidden_layers_size;
  for (int i = 1; i < nls; i++) {
    init_neural_layer(&nn->hidden_layers[i], nn->info.neurons_per[i], &nn->hidden_layers[i - 1], 0);
  }
  init_neural_layer(&nn->output_layer, nn->info.output_size, &nn->hidden_layers[nls - 1], 1);
}


void NN_forward_propagate(NN_neural_network_t *nn) {
  for (int i = 0; i < nn->info.hidden_layers_size; i++) {
    neural_layer_propagate(&nn->hidden_layers[i], nn->info.input_size);
  }
  neural_layer_propagate_regress(&nn->output_layer);
  for( int i = 0; i < nn->info.output_size; i++ )
    nn->prediction[i] = nn->output_layer.neurons[i].value;
}

// the effing meat and potatoes of this whol thing

void NN_backward_propagate(NN_neural_network_t *nn, double learning_rate ) {
  learning_rate = fabs( learning_rate);

  int output_size = nn->info.output_size;
  // calculate output layer errors and gradients
  NN_neural_layer_t *output_layer = &nn->output_layer;
  NN_neuron_t *output_neurons = output_layer->neurons;

  // ew, stack memory!
  double errors[NN_MAX_HIDDEN_LAYERS + 1][NN_MAX_NEURONS];

  double *output_error = errors[NN_MAX_HIDDEN_LAYERS];
  // compute output layer error
  for (int i = 0; i < output_size; i++) {
    double output = output_neurons[i].value;
    output_error[i] = (output - nn->target[i]) * tanh_deriv(output);
  }

  // update output layer weights and biases
  NN_neural_layer_t *last_hidden_layer = output_layer->feed;
  NN_neuron_t *last_hidden_neurons = last_hidden_layer->neurons;
  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < last_hidden_layer->size; j++) {
      output_neurons[i].weights[j] -= learning_rate * output_error[i] * last_hidden_neurons[j].value;
    }
    output_neurons[i].bias -= learning_rate * output_error[i];
  }

  NN_neural_layer_t *curr_layer = NULL;
  NN_neural_layer_t *next_layer = output_layer;

  for (int l = nn->info.hidden_layers_size - 1; l >= 0; l--) {
    curr_layer = &nn->hidden_layers[l];  //next_layer->feed
    NN_neuron_t *next_neurons = next_layer->neurons;

    // this layer's neuron 0 is scaled by next layer's weight 0
    // this layer's neuron 1 is scaled by next layer's weight 1
    // this layer's neuron 2 is scaled by next layer's weight 2
    // etc..
    // the ith neuron is fed into all the next layer's neurons, so we
    // iterate over all of the next layer neurons (using their respective weight)
    // and operate accordingly

    double *hidden_error = errors[l];
    double *hidden_error_next = errors[l + 1];

    // compute output layer error:
    for (int i = 0; i < curr_layer->size; i++) {
      hidden_error[i] = 0;
      for (int j = 0; j < next_layer->size; j++) {
        hidden_error[i] += hidden_error_next[j] * next_neurons[j].weights[i];
      }
      hidden_error[i] *= tanh_deriv(curr_layer->neurons[i].value);
    }

    // update weights and bias
    for (int i = 0; i < curr_layer->size; i++) {
      NN_neuron_t *neuron = &curr_layer->neurons[i];
      neuron->bias -= learning_rate * hidden_error[i];
      if (curr_layer->type > 0) {  //feed is another layer
        NN_neural_layer_t *prev_layer = curr_layer->feed;
        for (int j = 0; j < prev_layer->size; j++) {
          neuron->weights[j] -= learning_rate * hidden_error[i] * prev_layer->neurons[j].value;
        }
      } else if (curr_layer->type == 0) {

        for (int j = 0; j < nn->info.input_size; j++) {
          neuron->weights[j] -= learning_rate * hidden_error[i] * nn->input[j];
        }
      }

    }
    next_layer = curr_layer;
  }

}

void NN_train_neural_network( NN_neural_network_t *nn, double learning_rate ){
  learning_rate = fabs( learning_rate);
  NN_forward_propagate(nn);
  NN_backward_propagate(nn, learning_rate);
}

#pragma GCC diagnostic pop
