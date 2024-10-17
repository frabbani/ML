#pragma once

#include <cmath>

namespace nn {

constexpr int numEpochs = 100000;
constexpr double learningRate = 0.05;

struct Activation {
 public:
  enum Type {
    LeakyReLU,
    Sigmoid
  };
  virtual Type type() = 0;
  virtual double activation(double x) = 0;
  virtual double derivative(double x) = 0;
  virtual ~Activation() {
  }
};

struct LeakyReLUActivation : public Activation {
  Type type() override {
    return LeakyReLU;
  }
  double activation(double x) override {
    if (x < 0.0)
      return 0.01 * x;
    return 0.2 * x;
  }

  double derivative(double x) override {
    if (x < 0.0)
      return 0.01;
    return 0.2;
  }
};

struct SigmoidActivation : public Activation {
  Type type() override {
    return Sigmoid;
  }
  double activation(double x) override {
    return 1.0 / (1.0 + exp(-x));
  }

  double derivative(double x) override {
    return x * (1.0 - x);
  }
};

//struct TanhActivation : public Activation {
//  double activation(double x) override {
//    return tanh(x);
//  }
//  double derivative(double x) override {
//    x = tanh(x);
//    return 1.0 - x * x;
//  }
//};

struct HiddenLayer1 {
  struct Neuron {
    double weight = 1.0;
    double bias = 0.5;
    double value = 0.0;
  };
  const double &input;
  Neuron neuron1, neuron2;
  double error1 = 0.0, error2 = 0.0;
  HiddenLayer1(const double &input_)
      :
      input(input_) {
  }
  void forward(Activation *activation) {
    neuron1.value = neuron1.bias;
    neuron1.value += neuron1.weight * input;
    neuron1.value = activation->activation(neuron1.value);

    neuron2.value = neuron2.bias;
    neuron2.value += neuron2.weight * input;
    neuron2.value = activation->activation(neuron2.value);
  }

  void backPropagate() {
    neuron1.weight += learningRate * error1 * input;
    neuron1.bias += learningRate * error1;

    neuron2.weight += learningRate * error2 * input;
    neuron2.bias += learningRate * error2;
  }
};

struct HiddenLayer2 {
  struct Neuron {
    double weight1 = 0.0, weight2 = 0.0;
    double bias = 0.0;
    double value = 0.0;
  };

  const HiddenLayer1 &inputLayer;
  Neuron neuron1, neuron2;
  double error1 = 0.0, error2 = 0.0;

  HiddenLayer2(const HiddenLayer1 &layer)
      :
      inputLayer(layer) {
  }

  void forward(Activation *activation) {
    neuron1.value = neuron1.bias;
    neuron1.value += neuron1.weight1 * inputLayer.neuron1.value;
    neuron1.value += neuron1.weight2 * inputLayer.neuron2.value;
    neuron1.value = activation->activation(neuron1.value);

    neuron2.value = neuron2.bias;
    neuron2.value += neuron2.weight1 * inputLayer.neuron1.value;
    neuron2.value += neuron2.weight2 * inputLayer.neuron2.value;
    neuron2.value = activation->activation(neuron2.value);
  }

  void backPropagate() {
    neuron1.weight1 += learningRate * error1 * inputLayer.neuron1.value;
    neuron1.weight2 += learningRate * error1 * inputLayer.neuron2.value;
    neuron1.bias += learningRate * error1;

    neuron2.weight1 += learningRate * error2 * inputLayer.neuron1.value;
    neuron2.weight2 += learningRate * error2 * inputLayer.neuron2.value;
    neuron2.bias += learningRate * error2;
  }

};

struct OutputLayer {
  const HiddenLayer2 &inputLayer;
  double weight1 = 0.5, weight2 = 0.5, bias = 0.0;

  OutputLayer(const HiddenLayer2 &layer)
      :
      inputLayer(layer) {
  }

  double predict() const {
    return weight1 * inputLayer.neuron1.value
        + weight2 * inputLayer.neuron2.value + bias;
  }

};

struct NeuralNetwork {
  double input = 0.0;
  double prediction = 0.0;
  std::unique_ptr<Activation> activation = nullptr;

  OutputLayer predictionLayer;
  HiddenLayer2 hiddenLayer2;
  HiddenLayer1 hiddenLayer1;

  NeuralNetwork(Activation::Type activationType)
      :
      predictionLayer(hiddenLayer2),
      hiddenLayer2(hiddenLayer1),
      hiddenLayer1(input) {
    if( activationType == Activation::Type::Sigmoid )
      activation = std::make_unique<SigmoidActivation>();
    else if( activationType == Activation::Type::LeakyReLU ){
      activation = std::make_unique<LeakyReLUActivation>();
    }

  }

  double propagateForward() {
    hiddenLayer1.forward(activation.get());
    hiddenLayer2.forward(activation.get());
    return prediction = predictionLayer.predict();
  }

  void propagateBack(double target) {
    // forward propogation error
    double outputError = target - prediction;

    // output layer - 2 inputs, 1 output
    predictionLayer.weight1 += learningRate * outputError
        * hiddenLayer2.neuron1.value;
    predictionLayer.weight2 += learningRate * outputError
        * hiddenLayer2.neuron2.value;
    predictionLayer.bias += learningRate * outputError;

    // second hidden layer - 2 inputs, 2 outputs
    hiddenLayer2.error1 = outputError * predictionLayer.weight1
        * activation->derivative(hiddenLayer2.neuron1.value);

    hiddenLayer2.error2 = outputError * predictionLayer.weight2
        * activation->derivative(hiddenLayer2.neuron2.value);

    hiddenLayer2.backPropagate();

    // first hidden layer - 1 input, 2 outputs
    hiddenLayer1.error1 = 0.0;
    hiddenLayer1.error1 += hiddenLayer2.error1 * hiddenLayer2.neuron1.weight1;
    hiddenLayer1.error1 += hiddenLayer2.error2 * hiddenLayer2.neuron1.weight1;
    hiddenLayer1.error1 *= activation->derivative(hiddenLayer1.neuron1.value);

    hiddenLayer1.error2 = 0.0;
    hiddenLayer1.error2 += hiddenLayer2.error1 * hiddenLayer2.neuron2.weight1;
    hiddenLayer1.error2 += hiddenLayer2.error2 * hiddenLayer2.neuron2.weight1;
    hiddenLayer1.error2 *= activation->derivative(hiddenLayer1.neuron2.value);

    hiddenLayer1.backPropagate();
  }
};

}
