#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#include "neural.h"
#include "recurrent.h"

#define EPOCHS  500000
#define LEARNING_RATE 0.03
#define L2_LAMBDA 0.0001

void print_neural_layer(const NN_neural_layer_t *layer, int input_size) {
  char line[256];
  char token[32];

  for (int i = 0; i < layer->size; i++) {
    strcpy(line, "   w: ");
    const NN_neuron_t *neuron = &layer->neurons[i];
    int n = NN_first == layer->type ? input_size : layer->feed->size;
    for (int j = 0; j < n; j++) {
      sprintf(token, "%8.4lf", neuron->weights[j]);
      strcat(line, token);
    }
    strcat(line, " | b: ");
    sprintf(token, "%8.4lf\n", neuron->bias);
    strcat(line, token);
    printf("%s", line);
  }

}

void print_neural_network(NN_neural_network_t *nn) {
  printf("output layer:\n");
  print_neural_layer(&nn->output_layer, nn->info.input_size);
  for (int i = nn->info.hidden_layers_size - 1; i >= 0; i--) {
    printf("hidden layer %d (size %d):\n", i, nn->hidden_layers[i].size);
    print_neural_layer(&nn->hidden_layers[i], nn->info.input_size);
  }
}

double func(double x) {
  return 0.5 * x * x - 0.2;
}

void print_hidden_layer_rnn(RNN_neural_network_t *rnn, int layer_no) {
  if (layer_no >= rnn->info.hidden_layers_size)
    return;
  printf("***** LAYER %d *****\n", layer_no);
  RNN_neural_layer_t *layer = &rnn->hidden_layers[layer_no];
  for (int i = 0; i < layer->size; i++) {
    RNN_neuron_t *neuron = &layer->neurons[i];
    int nws = layer->type == NN_first ? rnn->info.input_size : layer->feed->size;
    printf("[");
    for (int j = 0; j < nws; j++)
      printf(" %+.6f", neuron->weights[j]);
    printf(" | %+.6f ]\n", neuron->bias);
  }

}

void testRNN(void) {
#define DEPTH   3
#define N       1000
#define LENGTH  (DEPTH * 5)
#undef EPOCHS
#define EPOCHS  250

  NN_seed_random(42);

  RNN_info_t info = { 0 };
  info.input_size = 1;
  info.output_size = 1;
  info.hidden_layers_size = 1;
  info.neurons_per[0] = 8;
  info.bptt_depth = DEPTH;
  info.learning_rate = 0.003;

  RNN_neural_network_t *rnn = malloc(sizeof *rnn);
  RNN_init_neural_network(rnn, &info);

  for (int e = 0; e < EPOCHS; ++e) {
    double mse = 0.0;
    int count = 0;
    double data[N][LENGTH];
    for (int i = 0; i < N; i++)
      for (int j = 0; j < LENGTH; j++)
        data[i][j] = NN_random(2.0, -1.0);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < LENGTH; j++) {
        mse += RNN_train_neural_network(rnn, &data[i][j], &data[i][(j - DEPTH + LENGTH) % LENGTH]);
        count++;
      }
    }
    printf("epoch %-3d | loss %.6f\n", e + 1, mse / (double) count);
  }

  printf("TEST:\n");
  double test[LENGTH];
  for (int i = 0; i < LENGTH; ++i)
    test[i] = NN_random(2.0, -1.0);

  for (int t = 0; t < LENGTH; ++t) {
    double targ = test[t - DEPTH];
    RNN_forward_propagate(rnn, &test[t], &targ);
    if (t < DEPTH)
      continue;
    double pred = rnn->prediction[0];
    printf("[%2d]  targ: %+.4f | pred: %+.4f\n", t, targ, pred);
  }
}

void testNN() {

  NN_neural_network_t *nn = malloc(sizeof(NN_neural_network_t));

  NN_info_t info;
  info.learning_rate = LEARNING_RATE;
  info.l2_decay = L2_LAMBDA;
  info.activation = NN_relu;
  info.hidden_layers_size = 2;
  info.input_size = 1;
  info.output_size = 1;
  for (int i = 0; i < info.hidden_layers_size; i++)
    info.neurons_per[i] = 15;

  NN_init_neural_network(nn, &info);

  printf("******************\n");
  print_neural_network(nn);
  printf("******************\n");

  for (int k = 0; k < EPOCHS; k++) {
    nn->input[0] = NN_random(2.0, -1.0);
    nn->target[0] = func(nn->input[0]);
    NN_train_neural_network(nn);
  }

  printf("******************\n");
  print_neural_network(nn);
  printf("******************\n");

  nn->input[0] = 0.123;
  NN_forward_propagate(nn);
  nn->target[0] = func(nn->input[0]);
  printf("input......: %lf\n", nn->input[0]);
  printf("target.....: %lf\n", nn->target[0]);
  printf("prediction.: %lf\n", nn->prediction[0]);
  printf("******************\n");

  double sum = 0.0f;
  for (int i = 0; i < 10; i++) {
    nn->input[0] = NN_random(2.0, -1.0);
    NN_forward_propagate(nn);
    nn->target[0] = func(nn->input[0]);

    double diff = (nn->target[0] - nn->prediction[0]);
    sum += diff * diff;
    printf("%d) target v. prediction: %lf v. %lf\n", i, nn->target[0], nn->prediction[0]);
  }
  printf("MSE: %lf", sum / 10.0);

  free(nn);
}

int main() {
  setbuf( stdout, NULL);
  printf("hello world!\n");
  testRNN();
  printf("goodbye!\n");
}
