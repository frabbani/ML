[LinkedIn Post](https://www.linkedin.com/feed/update/urn:li:activity:7252841572491276289?utm_source=share&utm_medium=member_desktop)

I’ve developed a simple neural network program that takes a single value as input and produces a single value as output. This straightforward network consists of two layers, each containing two neurons, to illustrate the learning process in action.

In this demonstration, I train the network to predict values for the exponential function “e^-5x”. To enhance the learning experience, I utilize two different activation functions: sigmoid and leaky ReLU.

The sigmoid activation function is named for its characteristic "S" shaped curve. It outputs values between 0 and 1, making it particularly useful for binary classification problems. The smoothness of the sigmoid curve helps stabilize learning, and in this demonstration, it creates a smooth graph that reflects the gradual nature of the learning process. However, the sigmoid function has its limitations.

In contrast, the leaky ReLU (Rectified Linear Unit) function introduces a linear output for positive values while allowing a small, non-zero contribution for negative values. This piecewise linear approach helps mitigate the vanishing gradient problem—a phenomenon where gradients diminish to near-zero values as they propagate through the network's layers. When gradients become very small, the network struggles to learn, potentially halting the learning process altogether.

The sigmoid activation function is particularly susceptible to this issue due to its "S" shape, which can lead to biased clumping of input values in the output range. This results in vanishing gradients, making it difficult for layers to update effectively, especially in deeper networks. 

In contrast to the gradual nature, the output from the leaky ReLU activation appears more segmented and linear, providing a different perspective on how the network learns. As you watch the animation clips, you’ll see how the learning process refines itself to closely match the target function, highlighting the impact of the chosen activation function on the network's performance.

I hope this demonstration sheds light on the fascinating dynamics of neural networks and the critical role activation functions play in shaping their learning capabilities.