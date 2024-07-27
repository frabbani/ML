#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#include "neural.h"

#define EPOCHS  1000000
#define LEARNING_RATE 0.0003


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
  return  0.5 * x - 0.2;
}

int main() {
  srand(time(NULL));
  setbuf( stdout, NULL);
  printf("hello world!\n");
  NN_neural_network_t *nn = malloc( sizeof(NN_neural_network_t));

  NN_info_t info;
  info.hidden_layers_size = 3;
  info.input_size = 1;
  info.output_size = 1;
  for( int i = 0; i < info.hidden_layers_size; i++ )
    info.neurons_per[i] = 3;


  NN_init_neural_network(nn, &info);

  printf("******************\n");
  print_neural_network(nn);
  printf("******************\n");

  for (int k = 0; k < 750000; k++) {
    nn->input[0] = NN_random( 2.0, -1.0 );
    nn->target[0] = func(nn->input[0]);
    NN_train_neural_network(nn, 0.05);
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
  for( int i = 0; i < 20; i++ ){
    nn->input[0] = NN_random( 2.0, -1.0 );
    NN_forward_propagate(nn);
    nn->target[0] = func(nn->input[0]);

    double diff = (nn->target[0] - nn->prediction[0]);
    sum += diff * diff;
    printf("target v. prediction: %lf v. %lf\n", nn->target[0], nn->prediction[0]);
  }
  printf( "MSE: %lf", sum / 20.0 );

  free( nn );
  return 0;


}
